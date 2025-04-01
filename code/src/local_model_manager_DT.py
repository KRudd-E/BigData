# local_model_manager_DT.py
"""
local_model_manager_DT.py

This file is responsible for training our local models.
- It will take the preprocessed Spark DataFrame and split it into partitions.
- For each partition, it trains a local model (like a proximity tree). - We do a DecisionTree for now.  
- Then, it collects all these local models into one ensemble.
- This ensemble will be used later for making predictions.
"""



from pyspark.sql import DataFrame
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import List
import logging

class LocalModelManager:
    """
    This class handles training local models on parts of our data and putting them in an ensemble.
    Steps:
      1. Get a preprocessed Spark DataFrame.
      2. Split it into partitions.
      3.  Train a simple model (like a decision tree) on each partition.
      4. Collect all models into one ensemble for later predictions.
    """

    def __init__(self, config: dict):
        """
        Initialize with settings from init .
        
        Args::
          config (dict): Settings like:
            - num_partitions: How many partitions to use.
            - model_params: Parameters for the model.
        """
        default_config = {
            "num_partitions": 2,
            "model_params": {"max_depth": 5},
            "model_params": {"random_state": 42}
        }
        self.config = default_config if config is None else config
        
        self.models: List[DecisionTreeClassifier] = []
        
        # Setup logger for our messages
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def train_ensemble(self, df: DataFrame) -> List[DecisionTreeClassifier]:
        """
        Train the ensemble of models.
        
        Args:
          df (DataFrame): A preprocessed Spark DataFrame with features and label.
        
        Returns:
          List[DecisionTreeClassifier]: A list of trained  models.
        """
        try:
            # Repartition the data if needed
            df = self._repartition_data(df)
            
            # Get number of partitions in the DataFrame
            num_partitions = df.rdd.getNumPartitions()
            self.logger.info(f"Training ensemble with {num_partitions} partitions....")
            
            # Train a model on each partition
            for partition_id in range(num_partitions):
                self.logger.debug(f"Processing partition {partition_id}")
                pandas_df = self._get_partition_data(df, partition_id)
                if pandas_df is not None:
                    model = self._train_partition_model(pandas_df)
                    self.models.append(model)
            
            self.logger.info(f"Successfully trained {len(self.models)} models")
            return self.models
            
        except Exception as e:
            self.logger.error(f"Error in train_ensemble: {str(e)}")
            raise

    def _repartition_data(self, df: DataFrame) -> DataFrame:
        """
        Repartition the data if "num_partitions" is set in the config.
        """
        if "num_partitions" in self.config:
            new_parts = self.config["num_partitions"]
            self.logger.info(f"Repartitioning data to {new_parts} partitions")
            return df.repartition(new_parts)
        return df

    def _get_partition_data(self, df: DataFrame, partition_id: int) -> pd.DataFrame:
        """
        Collect the rows from a specific partition and convert them into a pandas DataFrame.
        !!!!!!!!!!!!!!!!!!!! We use glom() to group the RDD by partitions, then grab the one we need.
        """
        try:
            # Get all partitions as a list of lists
            partitions = df.rdd.glom().collect()
            
            # !! Here we want to check if the partition exists!!
            if partition_id < 0 or partition_id >= len(partitions):
                self.logger.warning(f"Partition {partition_id} doesn't exist")
                return None
            
            # Get the rows in the requested partition
            rows = partitions[partition_id]
            if not rows:
                self.logger.warning(f"Partition {partition_id} is empty")
                return None
            
            # Convert each row to a dictionary and then to a pandas DataFrame
            return pd.DataFrame([row.asDict() for row in rows])
        
        except Exception as e:
            self.logger.error(f"Error in _get_partition_data: {str(e)}")
            return None


    def _train_partition_model(self, pandas_df: pd.DataFrame) -> DecisionTreeClassifier:
        """
        Train a simple decision tree model on a pandas DataFrame partition.
        """
        try:
            # Separate features (X) and label (y); assumes 'label' column is present.
            X = pandas_df.drop("label", axis=1)
            y = pandas_df["label"]
            
            # Create and train the model using any given parameters
            model = DecisionTreeClassifier(**self.config.get("model_params", {}))
            model.fit(X, y)
            return model
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def get_ensemble(self) -> List[DecisionTreeClassifier]:
        """
        Return the list of trained models.
        """
        return self.models



# from pyspark.sql import SparkSession
# # Create a Spark session
# spark = SparkSession.builder  \
#         .appName("RepartitionExample")  \
#         .master("local[*]")  \
#         .getOrCreate()

# # Create a sample DataFrame with 10 rows
# data = [(i, i * 2) for i in range(10)]
# df = spark.createDataFrame(data, ["id", "value"])

# # Print the initial number of partitions
# print("Initial number of partitions:", df.rdd.getNumPartitions())

# # Suppose our config says we want 3 partitions
# config = {"num_partitions": 3}

# # Repartition the DataFrame based on the config
# df_repart = df.repartition(config["num_partitions"])

# # Print the number of partitions after repartitioning
# print("Number of partitions after repartitioning:", df_repart.rdd.getNumPartitions())

# # Stop the session
# spark.stop()