# data_ingestion.py

"""
data_ingestion.py

This module shall handle loading raw data and returns a Spark DataFrame for subsequent stages of the pipeline.
"""

from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# THIS WAS HERE IN BEN'S DODGY COMMIT

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
        config.setdefault("data_percentage", 1.0)
        self.config = config


    def load_data(self):
        """Load data using file path from config."""
        data_path = self.config.get("data_path", "data/default.csv")  # Default path if not provided in config
        
        data_percentage = self.config.get("data_percentage", 1.0)
        
        print(f"Data Path: {data_path}")
        print(f"Loading {data_percentage*100}% of data")
        columns = [f"_c{i}" for i in range(1, 141)]
        
        # Create the schema correctly
        schema = StructType([
            StructField("label", IntegerType(), True)
        ] + [
            StructField(col, DoubleType(), True) for col in columns
        ])

        df = self.spark.read.csv(
            data_path, 
            header=True, 
            schema=schema, 
            sep=","
        )
        
        df = df.sample(fraction=data_percentage)     
        print(f"Data size: {df.count()}\n")
        self.validate_data(df)  # Check if data is valid.
        return df
    

    def validate_data(self, df):
        """Make sure the data is not empty."""
        # fetch a single row and check if the list is empty
        if len(df.take(1)) == 0:
            raise Exception("Data is empty!")

    def get_sample_data(self, df, fraction=0.3):
        """Return a small random sample of the data for testing."""
        return df.sample(fraction=fraction)