from pyspark.sql import SparkSession
from data_ingestion import DataIngestion

def main():
    # Create a local Spark session
    spark = SparkSession.getActiveSession()
    if spark is None:
        # No active session, so create a new one
        spark = SparkSession.builder \
            .appName("TestSparkInitialisation") \
            .master("local[*]") \
            .getOrCreate()
        print("Spark session created!")
    else:
        print("Spark session is already active!")

    #spark.sparkContext.setLogLevel("DEBUG")
       
    # Define the configuration with the correct path to the dataset.
    config = {
        "data_path": "D:/repos/BigData-main/BigData-1/ECG5000/*.tsv"       
    }
    
    # Create an instance of DataIngestion
    ingestion = DataIngestion(spark, config)
    
    # Load data
    df = ingestion.load_data()
    
    # Print schema and a few rows to confirm it loaded correctly
    print("Data schema:")
    df.printSchema()
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
