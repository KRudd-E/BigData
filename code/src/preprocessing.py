# preprocessing.py

"""
This module does some basic cleaning and transformations on a Spark DataFrame.
It comes after the data ingestion step.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

class Preprocessor:
    """
    This class cleans up our ECG data.
    It handles missing rows, splits the label from the features, and
    does a simple normalization on the feature columns.
    It returns a Spark DataFrame ready for training.
    """

    def __init__(self, config: dict):
        self.config = config

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """ Drops rows where every column is null """
        return df.dropna(how="all")

    def compute_min_max(self, df: DataFrame) -> dict:
        """
        Computes the min and max for each feature column (excluding 'label').
        """
        feature_cols = [col for col in df.columns if col != "label"]
        
        agg_exprs = []
        for col in feature_cols:
            agg_exprs.append(F.min(col).alias(f"min_{col}"))
            agg_exprs.append(F.max(col).alias(f"max_{col}"))
        
        stats_row = df.agg(*agg_exprs).first()
        
        min_max = {
            col: (stats_row[f"min_{col}"], stats_row[f"max_{col}"])
            for col in feature_cols
        }
        return min_max  

    # def normalize_features(self, df: DataFrame) -> DataFrame:
    #     """
    #     Applies a simple min-max scaling to all feature columns.
    #     I assume that after splitting, all columns except "label" are features.
    #     This is a basic normalization. We can replace it with a more robust method if needed.
    #     """
    #     feature_cols = [col for col in df.columns if col != "label"]

    #     # Compute min/max for all feature columns in one distributed aggregation
    #     agg_exprs = []
    #     for col in feature_cols:
    #         agg_exprs.append(F.min(col).alias(f"min_{col}"))
    #         agg_exprs.append(F.max(col).alias(f"max_{col}"))
        
    #     # Single aggregation + collect to driver
    #     stats_row = df.agg(*agg_exprs).first()
        
    #     # Extract min/max values into dictionaries
    #     min_vals = {col: stats_row[f"min_{col}"] for col in feature_cols}
    #     max_vals = {col: stats_row[f"max_{col}"] for col in feature_cols}
        
    #     # Define normalized columns in one distributed operation
    #     normalized_cols = []
    #     for col in feature_cols:
    #         min_val = min_vals[col]
    #         max_val = max_vals[col]
    #         if max_val != min_val:
    #             expr = (F.col(col) - min_val) / (max_val - min_val)
    #         else:
    #             expr = F.col(col)  # Avoid division by zero
    #         normalized_cols.append(expr.alias(col))
        
    #     # Apply all transformations in a single select (parallelized by Spark)
    #     return df.select(*normalized_cols, "label")
    
    
    def normalize(self, df: DataFrame, min_max: dict) -> DataFrame:
        """
        Normalizes feature columns using precomputed global min and max values.
        """
        feature_cols = [col for col in df.columns if col != "label"]
        
        normalized_cols = []
        for col in feature_cols:
            min_val, max_val = min_max[col]
            if max_val != min_val:
                expr = (F.col(col) - F.lit(min_val)) / (F.lit(max_val - min_val))
            else:
                expr = F.lit(0.0)  # if all values identical, set to zero or keep original
            normalized_cols.append(expr.alias(col))
        
        return df.select(*normalized_cols, "label")

    def _repartition_data_NotBalanced(self, df: DataFrame) -> DataFrame:
        if "num_partitions" in self.config:
            new_parts = self.config["num_partitions"]  # ✅ Get value first
            #self.logger.info(f"Repartitioning data to {new_parts} parts")
            return df.repartition(new_parts)
        return df
    
    def _repartition_data_Balanced(self, df: DataFrame, preserve_partition_id: bool = False) -> DataFrame:
        if "num_partitions" in self.config and "label_col" in self.config:
            num_parts = self.config["num_partitions"]
            label_col = self.config["label_col"]
            #self.logger.info(f"Stratified repartitioning into {num_parts} partitions")
            
            # Assign partition IDs (0 to num_parts-1 per class)
            # Subtracting 1 so that modulo is computed from 0
            window = Window.partitionBy(label_col).orderBy(F.rand())
            df = df.withColumn("_partition_id", ((F.row_number().over(window) - 1) % num_parts).cast("int"))
            
            # Force exact number of partitions using partition_id
            df = df.repartition(num_parts, F.col("_partition_id"))
            
            # For production, drop the helper column.
            if not preserve_partition_id:
                df = df.drop("_partition_id")
            return df
        return df

    def run_preprocessing(self, df: DataFrame) -> DataFrame:
        """
        Run all preprocessing steps in order:
         1. Drop rows that are completely null.
         2. Rename the first column as label.
         3. Normalize the feature columns.
        """
        
        df = self.handle_missing_values(df)
        min_max = self.compute_min_max(df) #for normalization
        
        df = self._repartition_data_Balanced(df)
        df = self.normalize(df, min_max)
        
        # You can add more feature engineering here if needed.
        return df
