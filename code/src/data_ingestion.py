# data_ingestion.py

"""
data_ingestion.py

This module shall handle loading raw data and returns a Spark DataFrame for subsequent stages of the pipeline.
"""

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame   



class DataIngestion:
    """
    Responsible for:       
        1. Loading data.
        2. Performing basic validations or sanity checks.
        3. Returning the loaded data in a Spark-dataframe format.
    """

    def __init__(self, spark: SparkSession, config: dict):
        """
        Set up Spark session and configuration.

        Args:
            spark (SparkSession): Spark session to be used for reading data.
            config (dict): Dictionary containing settings for data paths,
                           file formats, and any other parameters.
        """
        self.spark = spark
        self.config = config


    def load_data(self):
        """Load data using file path from config."""
        data_path = self.config.get("data_path", "data/default.csv")  # Default path if not provided in config
        # TODO: Add logic to handle default path 
        # Add logic to handle different file formats if needed.
        
        
        print(f"Data Path is {data_path} ++++++++++++++++++++++++++++++++++++++++++++++++")
        df = self.spark.read.option("header", "false")\
                    .option("delimiter", "\t")\
                    .option("inferSchema", "true")\
                    .csv(data_path)
                    
                   
        self.validate_data(df)  # Check if data is valid.
        return df
    

    def validate_data(self, df):
        """Make sure the data is not empty."""
        # Use limit(1) to fetch a single row and check if the list is empty
        if len(df.limit(1).collect()) == 0:
            raise Exception("Data is empty!")

    def get_sample_data(self, df, fraction=0.1):
        """Return a small random sample of the data for testing."""
        return df.sample(fraction=fraction)
