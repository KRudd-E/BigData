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
from prediction_manager import PredictionManager
from evaluation import Evaluator
from utilities import show_compact, randomSplit_dist
from visualization import plot_confusion_matrix, plot_class_metrics

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
        self.model_manager = None
        self.predictor = None
        self.evaluator = None

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
        if self.config["local_model_config"]["test_local_model"] is True:
            number_iterations = self.config["local_model_config"]["num_partitions"]
        else:
            number_iterations = self.config["global_model_config"]["num_partitions"]

        for i in range(self.config["min_number_iterarations"],number_iterations): #! iterations count
            """
            Run the whole pipeline:
            - Setup Spark session.
            - Load raw data.
            - Preprocess the data.
            - Train local models.
            - Generate predictions.
            - Evaluate the results.
            """
            self.config["local_model_config"]["num_partitions"] = i #! iterates over num partitions. 
            
            self._setup_spark()
            
            
            # Initialize modules
            self.evaluator = Evaluator()
            self.ingestion = DataIngestion(spark=self.spark, config=self.ingestion_config)
            self.preprocessor = Preprocessor(config=self.config)
            self.model_manager = LocalModelManager(config=self.config.get("local_model_config", None))
            #self.model_manager = GlobalModelManager(config=self.config.get("global_model_config", None))

            # Data Ingestion
            self.evaluator.start_timer("Ingestion")
            df = self.ingestion.load_data()
            self.evaluator.record_time("Ingestion")

            # Preprocessing
            self.evaluator.start_timer("Preprocessing")
            preprocessed_df = self.preprocessor.run_preprocessing(df)
            self.evaluator.record_time("Preprocessing")

            #print("\nSample of preprocessed data:")
            #show_compact(preprocessed_df, num_rows=5, num_cols=3)
            
            train_df, test_df = randomSplit_dist(preprocessed_df, weights=[0.8, 0.2], seed=123)

            # Training local models
            self.evaluator.start_timer("Training")
            ensemble = self.model_manager.train_ensemble(train_df)
            self.evaluator.record_time("Training")

            # Prediction
            self.evaluator.start_timer("Prediction")
            self.predictor = PredictionManager(self.spark, ensemble)
            predictions_df = self.predictor.generate_predictions(test_df)  
            self.evaluator.record_time("Prediction")
            
            print("\nPredictions:")
            predictions_df.groupBy("prediction").count().show() #! returns to driver

            # Evaluation
            report, class_names   = self.evaluator.log_metrics(predictions_df, ensemble=ensemble)
            
            # # Plot confusion matrix with proper labels
            # if "confusion_matrix" in report:
            #     plot_confusion_matrix(
            #         np.array(report["confusion_matrix"]),
            #         class_names,  
            #         save_path="pdf_results/confusion_matrix.pdf",
            #         show=False
            #     )
            
            # # Plot class metrics with proper labels
            # if "class_wise" in report:
            #     plot_class_metrics(
            #         report["class_wise"],
            #         class_names,  
            #         save_path="pdf_results/class_performance.pdf",
            #         show=False,
            #         class_names=class_names  # Pass to plotting function
            #     )

            # Save report 
            with open(f"code/src/reports/{i}.json", "w") as f: #! these will overwrite if run again. 
                json.dump(report, f, indent=2)
        
            print("\nFinal Report:")
            print(json.dumps(report, indent=2))

            # Clean up spark session if running locally
            if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
                self.spark.stop()
                print("\nLocal: Stopped Spark session!")
