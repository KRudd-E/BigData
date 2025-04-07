# prediction_manager.py
"""
Handles predictions using a trained ProximityForest model.
- Broadcasts the model to Spark workers
- Makes predictions on test data
- Returns predictions as a Spark DataFrame
"""

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
from aeon.classification.distance_based import ProximityForest
import numpy as np
import pandas as pd
import logging

class PredictionManager:
    def __init__(self, spark, ensemble: ProximityForest):
        """
        Initialize with a trained ProximityForest model.
        Args:
            spark: Spark session
            ensemble: Trained model from LocalModelManager.train_ensemble()
        
        """
        self.spark = spark
        self.ensemble = ensemble
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        
        # Basic model validation
        if not ensemble or not hasattr(ensemble, 'is_fitted') or not ensemble.is_fitted_:
            raise ValueError("Model is not trained. First call LocalModelManager.train_ensemble()")


    def _create_predict_udf(self):
        """Create Spark UDF for making predictions."""
        # Broadcast model to all workers
        broadcast_model = self.spark.sparkContext.broadcast(self.ensemble)
        
        @pandas_udf(DoubleType())
        def predict_udf(features: pd.Series) -> pd.Series:
            """Converts features to AEON format and makes predictions."""
            def predict_single(feature_array):
                try:
                    # Reshape to AEON's expected format: (samples, channels, features)
                    X = np.ascontiguousarray(feature_array).reshape(1, 1, -1)
                    return float(broadcast_model.value.predict(X)[0])
                except Exception as e:
                    print(f"Prediction error: {e}")
                    return float(-999)
  
            return features.apply(predict_single)
            
        return predict_udf

    def generate_predictions(self, test_df: DataFrame) -> DataFrame:
        
        """
        We take our test data and add a new column to it that will hold the predictions.        

        """
        # First, gotta make sure we actually have some models to use!
        if not self.ensemble:
            raise ValueError("No models available for prediction")

        feature_cols = [col for col in test_df.columns if col != "label"]
        
        test_df = test_df.withColumn(
            "features", 
            F.array(*[F.col(c).cast("double") for c in feature_cols])
        )
        
        predict_udf = self._create_predict_udf()
      
        predictions_df = test_df.withColumn(
            "prediction", 
            predict_udf("features")  
        ).drop("features")
        return predictions_df