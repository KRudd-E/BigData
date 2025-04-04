import os
from pyspark.sql import SparkSession
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from local_model_manager_DT import LocalModelManager_DT
from prediction_manager import PredictionManager
from local_model_manager import LocalModelManager
from utilities import show_compact
import time



def main():
    #======================== SET UP SPARK SESSION ========================
    
    # Check if we are in Databricks by looking for a special environment variable.
    # In Databricks, a Spark session (spark) is already available.
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        try:
            spark  # Try to use the existing 'spark' session.
            print("\nUsing Databricks Spark session.")
        except NameError:
            spark = SparkSession.builder.getOrCreate()
            print("\nDatabricks: Created Spark session!")
        config_DataIngestion = {
            "data_path": "/mnt/2025-team6/fulldataset_ECG5000.csv",
            "data_percentage": 0.005  # % Percentage of data to load for SW development
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
            print("\nLocal: Created Spark session!")
        else:
            print("\nLocal: Spark session already active!")

        # For local run, use your Windows file path.
    
        # Make sure project root folder is the parent of "code" folder, <BigData>
        print("Current working directory (project root):", os.getcwd())
        # Get the directory where the current file (main.py) is located
        current_dir = os.path.dirname(__file__)
        # Go up two levels to reach the project root
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

        config_DataIngestion = {
            # "data_path": project_root + "/ECG5000/*.tsv" # IF WE DECIDE TO USE .tsv FILES
            "data_path": project_root + "/fulldataset_ECG5000.csv", # IF WE DECIDE TO USE .csv FILES
            "data_percentage": 0.005  # % Percentage of data to load for SW development
            }   
    
    start = time.time()
    #===================================== LOAD DATA =====================================
    # Create an instance of DataIngestion and load the data.
    ingestion = DataIngestion(spark, config_DataIngestion)
    df = ingestion.load_data()

    #df.printSchema()
    # print("\nSample of raw data:")
    # show_compact(df, num_rows=5, num_cols=3)

    #=================================== PREPROCESS DATA ==================================
    # Create an instance of Preprocessor and run all cleaning steps.
    config_preproc = {}
    preprocessor = Preprocessor(config_preproc)
    preprocessed_df = preprocessor.run_preprocessing(df)
    data_load_time = time.time()
    print(f"\nData load + preprocessing time: {data_load_time-start:.4f} seconds\n")
    
    print("Sample of preprocessed data:")
    show_compact(preprocessed_df, num_rows=5, num_cols=3)
    
    #==================================== TRAIN DIFFERENT MODELS ===========================
    # Train local models on the preprocessed data
    

    #************************************** PF- using AEON ********************************
    start = time.time()
    print("\nHere we train Proximity forests on the preprocessed data.")
    model_manager = LocalModelManager({"num_partitions": 2, "model_params": {"random_state": 42}})
    ensemble = model_manager.train_ensemble(preprocessed_df)
    training_time = time.time()
    print(f"Training time: {training_time-start:.4f} seconds")

    #============================ TEST DIFFERENT MODELS ======================================
    start = time.time()
    predictor = PredictionManager(spark, ensemble)
    predictions_df = predictor.generate_predictions(preprocessed_df)
    prediction_time = time.time()
    print(f"Prediction time: {prediction_time-start:.4f} seconds")
    
    #============================= EVALUATE DIFFERENT MODELS =================================
    # If running locally, stop the Spark session.
    # In Databricks, do not stop the session as it is managed by the platform.
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        spark.stop()

        print("\nLocal: Stopped Spark session!")

if __name__ == "__main__":
    print("Running main.py")
    print("Current working directory (project root):", os.getcwd())
    main()#