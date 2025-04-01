# local_model_manager.py
"""
local_model_manager.py

This file is in charge of training our local models.
It takes the preprocessed Spark DataFrame and splits it into parts.
For each part, it trains a Proximity Forest model.
Then, it gathers all the models into one ensemble that we can use later to make predictions.
"""

import numpy as np
from pyspark.sql import DataFrame
import pandas as pd
from aeon.classification.distance_based import ProximityForest  # using ProximityForest from aeon
from typing import List
import logging

class LocalModelManager:
    """
    This class handles training local models on chunks of our data and then
    puts them together into an ensemble.
    
    The steps are pretty simple:
      1. Get a preprocessed Spark DataFrame.
      2. Split it into parts.
      3. Train a Proximity Forest model on each part.
      4. Collect all the models so we can use them later for predictions.
    """

    def __init__(self, config: dict):
        """
        Init with our settings.
        
        Args:
            config (dict): Settings like:
              - num_partitions: How many parts to split the data into.
              - model_params: Extra parameters for the Proximity Forest model.
        """
        # Use default settings if none provided
        default_config = {
            "num_partitions": 2,
            "model_params": {"random_state": 42}
        }
        self.config = default_config if config is None else config
        self.models: List[ProximityForest] = []
        
        # Set up a logger so we can see whats going on
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def train_ensemble(self, df: DataFrame) -> List[ProximityForest]:
        """
        Main function to train our ensemble of models.
        
        Args:
            df (DataFrame): A preprocessed Spark DataFrame with features and a label.
        
        Returns:
            List[ProximityForest]: A list of our trained Proximity Forest models.
        """
        try:
            # Repartition the data if our config says so
            df = self._repartition_data(df)
            
            # Get the number of parts we have now
            num_parts = df.rdd.getNumPartitions()
            self.logger.info(f"Training ensemble with {num_parts} parts...")
            
            # For each part, train a model
            for part_id in range(num_parts):
                self.logger.debug(f"Working on part {part_id}")
                pandas_df = self._get_partition_data(df, part_id)
                if pandas_df is not None:
                    model = self._train_partition_model(pandas_df)
                    self.models.append(model)
            
            self.logger.info(f"Trained {len(self.models)} models successfully")
            return self.models
            
        except Exception as e:
            self.logger.error(f"Error in train_ensemble: {str(e)}")
            raise

    def _repartition_data(self, df: DataFrame) -> DataFrame:
        """
        If 'num_partitions' is set in the config, repartition the data accordingly.
        """
        if "num_partitions" in self.config:
            new_parts = self.config["num_partitions"]
            self.logger.info(f"Repartitioning data to {new_parts} parts")
            return df.repartition(new_parts)
        return df

    def _get_partition_data(self, df: DataFrame, partition_id: int) -> pd.DataFrame:
        """
        Get the rows from one part and convert them to a pandas DataFrame.
        We use glom() to group the RDD by parts, then pick the one we need.
        """
        try:
            parts = df.rdd.glom().collect()
            if partition_id < 0 or partition_id >= len(parts):
                self.logger.warning(f"Part {partition_id} does not exist")
                return None
            rows = parts[partition_id]
            if not rows:
                self.logger.warning(f"Part {partition_id} is empty")
                return None
            return pd.DataFrame([row.asDict() for row in rows])
        except Exception as e:
            self.logger.error(f"Error in _get_partition_data: {str(e)}")
            return None

    def _train_partition_model(self, pandas_df: pd.DataFrame) -> ProximityForest:
        """
        Train a Proximity Forest model on one pandas DataFrame part.
        """
        try:
            # Split the data into features (X) and label (y). Assumes 'label' col exists.
            X = pandas_df.drop("label", axis=1)
            y = pandas_df["label"]
            
            # Convert to contiguous numpy arrays to avoid reshaping errors
            X = np.ascontiguousarray(X.values)
            y = np.ascontiguousarray(y.values)
            
            # Create and train the Proximity Forest model using given params
            model = ProximityForest(**self.config.get("model_params", {}))
            model.fit(X, y)
            return model
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def get_ensemble(self) -> List[ProximityForest]:
        """
        Return the list of our trained models.
        """
        return self.models
