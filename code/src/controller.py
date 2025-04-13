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
from pyspark.sql import SparkSession
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from local_model_manager import LocalModelManager
from prediction_manager import PredictionManager
from evaluation import Evaluator
from utilities import show_compact

class PipelineController:
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
        """
        Run the whole pipeline:
          - Setup Spark session.
          - Load raw data.
          - Preprocess the data.
          - Train local models.
          - Generate predictions.
          - Evaluate the results.
        """
        self._setup_spark()

        # Initialize modules
        self.evaluator = Evaluator()
        self.ingestion = DataIngestion(self.spark, self.ingestion_config)
        self.preprocessor = Preprocessor(config={})
        self.model_manager = LocalModelManager(config=self.config.get("local_model_config", None))
          

        # Data Ingestion
        self.evaluator.start_timer("ingestion")
        df = self.ingestion.load_data()
        self.evaluator.record_time("ingestion")

        # Preprocessing
        self.evaluator.start_timer("preprocessing")
        preprocessed_df = self.preprocessor.run_preprocessing(df)
        self.evaluator.record_time("preprocessing")

        print("\nSample of preprocessed data:")
        show_compact(preprocessed_df, num_rows=5, num_cols=3)

        # Training local models
        self.evaluator.start_timer("training")
        ensemble = self.model_manager.train_ensemble(preprocessed_df)
        self.evaluator.record_time("training")


        # Prediction
        self.predictor = PredictionManager(self.spark, ensemble)
        
        self.evaluator.start_timer("prediction")
        predictions_df = self.predictor.generate_predictions(preprocessed_df)  # Using preprocessed data as test here
        self.evaluator.record_time("prediction")
        
        print("\nPredictions:")
        predictions_df.groupBy("prediction").count().show()

        # Evaluation
        report = self.evaluator.log_metrics(predictions_df, ensemble=ensemble)
        print("\nFinal Report:")
        print(json.dumps(report, indent=2))

        # Clean up spark session if running locally
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            self.spark.stop()
            print("\nLocal: Stopped Spark session!")


if __name__ == "__main__":
    print("Starting pipeline via controller.py")
    # Example config; adjust paths and parameters as needed.
    config = {
        "databricks_data_path": "/mnt/2025-team6/fulldataset_ECG5000.csv",
        "local_data_path": "/fulldataset_ECG5000.csv",
        "data_percentage": 0.5,
        "local_model_config": {
            "num_partitions": 2, 
            "model_params": {
                "random_state": 1234
                }
            }
    }
    controller = PipelineController(config)
    controller.run()