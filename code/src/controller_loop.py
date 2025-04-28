# controller.py
"""
controller.py

This file ties the whole pipeline together.
Once data ingestion, preprocessing, model training, prediction, and evaluation are in place,
this controller calls each step in the right order.
It acts as the main coordinator of the process.
"""

import os
import json
import numpy as np
from pyspark.sql import SparkSession
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from local_model_manager import LocalModelManager
from global_model_manager import GlobalModelManager
from prediction_manager import PredictionManager, predict_with_global_prox_tree
from evaluation import Evaluator
from utilities import show_compact, randomSplit_dist, randomSplit_stratified_via_sampleBy, compute_min_max
from visualization import plot_confusion_matrix, plot_class_metrics
import logging
import time

class PipelineController_Loop:
    def __init__(self, config):
        """
        Initialize the controller with our configuration.
        """
        self.config = config
        self.spark = None
        self.ingestion_config = {}
        self.ingestion = None
        self.preprocessor = None
        self.local_model_manager = None
        self.global_model_manager = None
        self.predictor = None
        self.evaluator = None
        
        # Set up a logger so we can see whats going on
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def _setup_spark(self):
        """
        Setup Spark session based on the environment.
        """
        # Check for Databricks environment
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            self.spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            print("\nUsing Databricks Spark session.")
            # Use the Databricks file path from config.
            self.ingestion_config = {
                "data_path": self.config.get("databricks_data_path", "/mnt/2025-team6/fulldataset_ECG5000.csv"),
                "data_percentage": self.config.get("data_percentage", 0.05)
            }
        else:
            self.spark = SparkSession.getActiveSession() or SparkSession.builder \
                .appName("LocalPipeline") \
                .master("local[*]") \
                .getOrCreate()
            print("\nUsing local Spark session.")
            # Set up a local file path, using the project root folder.
            current_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.ingestion_config = {
                "data_path": project_root + self.config.get("local_data_path", "/fulldataset_ECG5000.csv"),
                "data_percentage": self.config.get("data_percentage", 0.5)
            }
            print("Current working directory (project root):", os.getcwd())

    def run(self):
        if self.config["model_type"] == "local":
            number_iterations = self.config["local_model_config"]["num_partitions"]
        elif self.config["model_type"] == "global":
            number_iterations = self.config["global_model_config"]["num_partitions"]
        else:
            raise ValueError("Invalid model type. Choose 'local' or 'global'.")

        for i in range(self.config["min_number_iterarations"], number_iterations + 1 ): #! iterations count from min to num_partitions inclusive.
            """
            Run the whole pipeline:
            - Setup Spark session.
            - Load raw data.
            - Preprocess the data.
            - Train local models.
            - Generate predictions.
            - Evaluate the results.
            """
            if self.config["local_model_config"]["test_local_model"] is True:
                self.config["local_model_config"]["num_partitions"] = i #! iterates over num partitions for global model. 
                self.logger.info(f"\n\nRunning local model with {i} partitions")
                
            if self.config["global_model_config"]["test_global_model"] is True:
                self.config["global_model_config"]["num_partitions"] = i #! iterates over num partitions for global model.
                self.logger.info(f"\n\nRunning global model with {i} partitions")
            
            self._setup_spark()
            current_datetime = time.strftime("%Y-%m-%d-%H-%M")
        
            try:
                module_path = os.path.join(os.getcwd(), 'code', 'src', 'global_model_manager.py')
                self.spark.sparkContext.addPyFile(module_path)
                self.logger.info(f"Added {module_path} to SparkContext pyFiles.")
            except Exception as e:
                self.logger.error(f"Failed to add global_model_manager.py to SparkContext: {e}")    
            # Handle error appropriately, maybe raise it
            
            # Initialize modules
            self.evaluator = Evaluator()
            self.ingestion = DataIngestion(spark=self.spark, config=self.ingestion_config)
            self.preprocessor = Preprocessor(config=self.config)
            
  
            # Data Ingestion
            self.evaluator.start_timer("Ingestion")
            df = self.ingestion.load_data()
            self.evaluator.record_time("Ingestion")

            #  Split data first - does one shuffle of the data
            min_max = compute_min_max(df) # we should save this as artefact for later use.
            train_df, test_df = randomSplit_stratified_via_sampleBy(df, label_col = "label", weights=[0.8, 0.2], seed=123)
            #train_df, test_df = randomSplit_dist(preprocessed_df,  weights=[0.8, 0.2], seed=123)          
            
            # Preprocessing train data
            self.evaluator.start_timer("Preprocessing train data")
            preprocessed_train_df = self.preprocessor.run_preprocessing(train_df, min_max) # does two shuffles of the 80% of data 
            self.evaluator.record_time("Preprocessing train data")

            # Preprocessing test data
            self.evaluator.start_timer("Preprocessing test data")
            preprocessed_test_df = self.preprocessor.run_preprocessing(test_df, min_max) # does two shuffles of the rest of 20% of data 
            self.evaluator.record_time("Preprocessing test data")
            
            # print("\nSample of preprocessed data:")
            # show_compact(preprocessed_train_df, num_rows=5, num_cols=3)                   
            
            # Training global models
            if self.config["global_model_config"]["test_global_model"] is True:
                self.global_model_manager = GlobalModelManager(spark=self.spark, config=self.config.get("global_model_config", None))
                self.evaluator.start_timer("Training Global Models")
                model_ensamble = self.global_model_manager.fit(preprocessed_train_df)
                self.evaluator.record_time("Training Global Models")
                
                # Prediction
                self.evaluator.start_timer("Prediction global models")
                
                predictions_df = predict_with_global_prox_tree(model_ensamble, preprocessed_test_df)

                self.evaluator.record_time("Prediction global models")
                
                print("\nLocal model Predictions:")
                predictions_df.groupBy("prediction").count().show() #! returns to driver

                # Evaluation
                report, class_names   = self.evaluator.log_metrics(predictions_df, model=model_ensamble)
                
                  # Load existing data if file exists, else create empty dict
                if os.path.exists(f"code/src/logs/report_global_model_{current_datetime}.json"):
                    with open(f"code/src/logs/report_global_model_{current_datetime}.json", "r") as f:
                        all_reports = json.load(f)
                else:
                    all_reports = {}

                # Add new report under key i (cast to str for JSON compatibility)
                all_reports[str(i)] = report

                # Write full dictionary back to file
                with open(f"code/src/logs/report_global_model_{current_datetime}.json", "w") as f:
                    json.dump(all_reports, f, indent=2)
                
                print("\nFinal Report global model:")
                print(json.dumps(report, indent=2))

                delay_seconds = 10 
                print(f"\n\nWaiting for {delay_seconds} seconds before next iteration...\n")
                time.sleep(delay_seconds)
            
            # Training local models
            if self.config["local_model_config"]["test_local_model"] is True:
                self.local_model_manager = LocalModelManager(config=self.config.get("local_model_config", None))
                self.evaluator.start_timer("Training Local Models")
                model_ensamble = self.local_model_manager.train_ensemble(preprocessed_train_df)
                self.evaluator.record_time("Training Local Models")
                
                # Prediction
                self.evaluator.start_timer("Prediction local models")
                self.predictor = PredictionManager(self.spark, model_ensamble)
                predictions_df = self.predictor._generate_predictions_local(preprocessed_test_df)  
                self.evaluator.record_time("Prediction local models")
            
                print("\nLocal model Predictions:")
                predictions_df.groupBy("prediction").count().show() #! returns to driver

                # Evaluation
                report, class_names   = self.evaluator.log_metrics(predictions_df, model=model_ensamble)

                
                # Load existing data if file exists, else create empty dict
                if os.path.exists(f"code/src/logs/report_local_model_{current_datetime}.json"):
                    with open(f"code/src/logs/report_local_model_{current_datetime}.json", "r") as f:
                        all_reports = json.load(f)
                else:
                    all_reports = {}

                # Add new report under key i (cast to str for JSON compatibility)
                all_reports[str(i)] = report

                # Write full dictionary back to file
                with open(f"code/src/logs/report_local_model_{current_datetime}.json", "w") as f:
                    json.dump(all_reports, f, indent=2)
                
                print("\nFinal Report Local model:")
                print(json.dumps(report, indent=2))




            # Clean up spark session if running locally
            if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
                self.spark.stop()
                print("\nLocal: Stopped Spark session!")