from pyspark.sql import SparkSession
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from utilities import show_compact

print("Running main.py")

def main():
    # Create a local Spark session
    spark = SparkSession.getActiveSession()
    if spark is None:
        # No active session, so create a new one
        spark = SparkSession.builder \
            .appName("TestPipeline") \
            .master("local[*]") \
            .getOrCreate()
        print("Spark session created!")
    else:
        print("Spark session is already active!")

    # Define the configuration with the local path to the dataset.
    config = {
        "data_path": 'train+test_ECG5000.tsv'
    }

    # Create an instance of DataIngestion
    ingestion = DataIngestion(spark, config)
    
    # Load data
    df = ingestion.load_data()
    
    # Print schema and a few rows to confirm it loaded correctly
    print("Data schema:")
    df.printSchema()
    
    # Instantiate the preprocessor (config can be expanded later!!)
    config_preproc = {}
    preprocessor = Preprocessor(config_preproc)
    preprocessed_df = preprocessor.run_preprocessing(df)
    
    print("Sample of raw data")
    show_compact(df, num_rows=5, num_cols=3)
    print("Sample of preprocessed data:")
    show_compact(preprocessed_df, num_rows=5, num_cols=3)  
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
