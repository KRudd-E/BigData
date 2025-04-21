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
   
    
    def normalize(self, df: DataFrame, min_max: dict, preserve_partition_id: bool = False) -> DataFrame:
        """
        Normalizes feature columns using precomputed global min and max values.
        """
        feature_cols = [col for col in df.columns if (col != "label" and col != "_partition_id")]
        
        normalized_cols = []
        for col in feature_cols:
            min_val, max_val = min_max[col]
            if max_val != min_val:
                expr = (F.col(col) - F.lit(min_val)) / (F.lit(max_val - min_val))
            else:
                expr = F.lit(0.0)  # if all values identical, set to zero or keep original
            normalized_cols.append(expr.alias(col))
        
        return df.select(*normalized_cols, "label", "_partition_id")

    def _repartition_data_NotBalanced(self, df: DataFrame) -> DataFrame:
        if "num_partitions" in self.config:
            new_parts = self.config["num_partitions"]  # 
            #self.logger.info(f"Repartitioning data to {new_parts} parts")
            return df.repartition(new_parts)
        return df
    
    def _repartition_data_Balanced(self, df: DataFrame, preserve_partition_id: bool = False) -> DataFrame:
        
        if ("num_partitions" in self.config["local_model_config"] \
            or "num_partitions" in self.config["global_model_config"]) \
            and "label_col" in self.config:
            
            if self.config["local_model_config"]["test_local_model"] is True:
                num_parts = self.config["local_model_config"]["num_partitions"]
            else:
                num_parts = self.config["global_model_config"]["num_partitions"]
                
            label_col = self.config["label_col"]
            # self.logger.info(f"Stratified repartitioning into {num_parts} partitions")
            
            # Assign partition IDs (0 to num_parts-1 per class)
            # Subtracting 1 so that modulo is computed from 0
            window = Window.partitionBy(label_col) \
                            .orderBy(F.rand())          #  one shuffles to group all rows of each label together so we can number them
                            
            df = df.withColumn("_partition_id", ((F.row_number().over(window) - 1) % num_parts).cast("int"))
            
            # Force exact number of partitions using partition_id
            df = df.repartition(num_parts, F.col("_partition_id"))          # one shuffles to repartition by _partition_id to ensure we have num_parts partitions
            print(f'Repartitioning to <<<< {num_parts} >>>> workers - partitions.')
            
            if not preserve_partition_id:
                df = df.drop("_partition_id")
            return df
            
        return df

    def run_preprocessing(self, df: DataFrame, min_max) -> DataFrame:
        """
        Args:
            df (DataFrame): Input DataFrame to be preprocessed. pyspark Sql DataFrame.
        Returns:
            DataFrame: Preprocessed DataFrame ready for training.
        
            
        Run all preprocessing steps in order:
         1. Drop rows that are completely null.
         2. Repartition the data : shuffle the data to balance the partitions.
         3. Normalize the feature columns.
        """
        df = self.handle_missing_values(df)        
        df = self._repartition_data_Balanced(df, preserve_partition_id = self.config["reserve_partition_id"]) # data is  redistributed across partitions -> 2 shuffle
        df = self.normalize(df, min_max, preserve_partition_id = self.config["reserve_partition_id"])   # data is normalised on partitions -> no shuffle
        
        # You can add more feature engineering here if needed.
        return df
