# preprocessing.py

"""
This module does some basic cleaning and transformations on a Spark DataFrame.
It comes after the data ingestion step.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class Preprocessor:
    """
    This class cleans up our ECG data.
    It handles missing rows, splits the label from the features, and
    does a simple normalization on the feature columns.
    It returns a Spark DataFrame ready for training.
    """

    def __init__(self, config: dict):
        """
        Set up with our simple settings.
        
        Args:
            config (dict): A dict that can include options like which columns to normalize.
                         (Not used here noe.... but can be extended later.)
        """
        self.config = config

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Drops rows where every column is null.
        """
        return df.dropna(how="all")

    def normalize_features(self, df: DataFrame) -> DataFrame:
        """
        Applies a simple min-max scaling to all feature columns.
        I assume that after splitting, all columns except "label" are features.
        This is a basic normalization. We can replace it with a more robust method if needed.
        """
        feature_cols = [col for col in df.columns if col != "label"]
        for col in feature_cols:
            # Get min and max for the column
            stats = df.select(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
            min_val, max_val = stats["min"], stats["max"]
            if max_val is not None and max_val != min_val:
                df = df.withColumn(col, (F.col(col) - min_val) / (max_val - min_val))
        return df

    def run_preprocessing(self, df: DataFrame) -> DataFrame:
        """
        Run all preprocessing steps in order:
         1. Drop rows that are completely null.
         2. Rename the first column as label.
         3. Normalize the feature columns.
        """
        df = self.handle_missing_values(df)
        df = self.normalize_features(df)
        # You can add more feature engineering here if needed.
        return df
