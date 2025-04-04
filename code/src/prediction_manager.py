# prediction_manager.py
"""
prediction_manager.py

This file handles making predictions using our trained models.
- It takes the ensemble of local models and distributes them to worker nodes.
- It applies the models to test data to generate predictions.
- Finally, it collects the predictions back at the driver for analysis.
"""
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import IntegerType
from pyspark import TaskContext
from typing import List
import numpy as np
from aeon.classification.distance_based import ProximityForest
import pandas as pd
from pyspark.sql.functions import pandas_udf
import logging

class PredictionManager:
    """
    we use this to set up the prediction manager with our models and settings.
    Takes the bunch of models we have and sends these models to executors for parallel inference.
    For each piece of data, it gets a prediction from each model.
    Use majority voting to aggregate predictions.
    """

    def __init__(self, spark, models: List[ProximityForest]):
    
        """
        When we set up the PredictionManager, we need to give it the Spark session
        and the list of models we want to use.
        
        **** We also set up a logger to see if anything goes wrong.
        """
        self.spark = spark
        self.models = models
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def _predict_udf(self):
        """ We need a way to send our models to all the workers in our Spark cluster. 
        We do this by 'broadcasting' them, then this function creates a UDF 
        that each worker can use to make predictions with these models.
        """
        broadcast_models = self.spark.sparkContext.broadcast(self.models)
        
        @pandas_udf(IntegerType())
        def _predict(features: pd.Series) -> pd.Series:
            try:
                    X = np.array(x).reshape(1, -1)
                    preds = [model.predict(X)[0] for model in broadcast_models.value]
                    return np.argmax(np.bincount(preds))
            except Exception as e:
                    print(f"Error in partition {TaskContext.get().partitionId()}: {str(e)}")
                    return -1  # Error code
            return features.apply(predict_single)
        
        return _predict


    def generate_predictions(self, test_df: DataFrame) -> DataFrame:
        
        """
        We take our test data and add a new column to it that will hold the predictions.        

        """
        # First, gotta make sure we actually have some models to use!
        if not self.models:
            raise ValueError("No models available for prediction")

        feature_cols = [col for col in test_df.columns if col != "label"]
        
        test_df = test_df.withColumn(
            "features", 
            F.array(*[F.col(c).cast("double") for c in feature_cols])
        )
        
        predict_udf = self._predict_udf()
        predictions_df = test_df.withColumn(
            "prediction", 
            predict_udf("features")
        ).drop("features")
        
        return predictions_df