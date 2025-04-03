from pyspark.sql import SparkSession
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from local_model_manager_DT import LocalModelManager_DT
# from local_model_manager import LocalModelManager
from utilities import show_compact
import os

def main():
    # Check if we are in Databricks by looking for a special environment variable.
    # In Databricks, a Spark session (spark) is already available.
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        try:
            spark  # Try to use the existing 'spark' session.
            print("Using Databricks Spark session.")
        except NameError:
            spark = SparkSession.builder.getOrCreate()
            print("Databricks: Created Spark session!")
        config = {
            "data_path": "/mnt/2025-team6/*.tsv"
        }
    else:
        # If not in Databricks, we are running locally.
        # Try to get an active Spark session; if none, create one.
        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = SparkSession.builder \
                .appName("TestPipeline") \
                .master("local[*]") \
                .getOrCreate()
            print("Local: Created Spark session!")
        else:
            print("Local: Spark session already active!")
        
    # For local run, use your Windows file path.
    
    # Make sure project root folder is the parent of "code" folder, <BigData>
    print("Current working directory (project root):", os.getcwd())
    # Get the directory where the current file (main.py) is located
    current_dir = os.path.dirname(__file__)
    # Go up two levels to reach the project root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    config = {
        "data_path": project_root + "/ECG5000/*.tsv"
    }
    # Create an instance of DataIngestion and load the data.
    ingestion = DataIngestion(spark, config)
    df = ingestion.load_data()

    print("Raw Data schema:")
    df.printSchema()

    # Create an instance of Preprocessor and run all cleaning steps.
    config_preproc = {}
    preprocessor = Preprocessor(config_preproc)
    preprocessed_df = preprocessor.run_preprocessing(df)
    
    # Train local models on the preprocessed data
    
    # Here we train DecisionTree on the preprocessed data.
    print("Here we train Decision Tree on the preprocessed data.")
    config_model = {
        "num_partitions": 2, 
        "model_params": {"random_state": 42}
    }
    model_manager_DT = LocalModelManager_DT(config_model)
    ensemble_DT = model_manager_DT.train_ensemble(preprocessed_df)
    
    # # Here we train Proximity forests on the preprocessed data.
    # print("Here we train Proximity forests on the preprocessed data.")
    # model_manager = LocalModelManager({"num_partitions": 2, "model_params": {"random_state": 42}})
    # ensemble = model_manager.train_ensemble(preprocessed_df)

    print("Sample of raw data:")
    show_compact(df, num_rows=5, num_cols=3)
    print("Sample of preprocessed data:")
    show_compact(preprocessed_df, num_rows=5, num_cols=3)

    # If running locally, stop the Spark session.
    # In Databricks, do not stop the session as it is managed by the platform.
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        spark.stop()

if __name__ == "__main__":
    print("Current working directory (project root):", os.getcwd())
    main()#