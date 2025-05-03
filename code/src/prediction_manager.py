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


# Set up a logger for this external function if needed
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def predict_with_global_prox_tree(global_tree_model, data_df: DataFrame) -> DataFrame:

    logger.info("Starting external prediction with GlobalProxTree.")
    



    if not (hasattr(global_tree_model, 'predict') and callable(global_tree_model.predict) and hasattr(global_tree_model, 'spark')):
        model_type_name = type(global_tree_model).__name__ if global_tree_model is not None else "None"
        logger.error(f"Invalid model type provided. Expected GlobalProxTree-like object, got {model_type_name}.")
        raise TypeError(f"Invalid model type provided to predict_with_global_prox_tree. Expected GlobalProxTree-like object.")

    predictions_df = global_tree_model.predict(data_df)

    

        # --- Ensure the output DataFrame has a 'label' column for evaluation ---
    # The GlobalProxTree.predict method returns 'true_label' and 'prediction'.
    # The Evaluator expects 'label' and 'prediction'.
    # Rename 'true_label' to 'label' if it exists and 'label' does not.
    if "true_label" in predictions_df.columns and "label" not in predictions_df.columns:
        logger.debug("Renaming 'true_label' to 'label' in predictions DataFrame for evaluation compatibility.")
        predictions_df = predictions_df.withColumnRenamed("true_label", "label")
    # If 'label' already exists, no renaming is needed.
    # If neither exists, the Evaluator will log a warning.

    # Ensure 'prediction' column exists (should be returned by GlobalProxTree.predict)
    if "prediction" not in predictions_df.columns:
         logger.error("GlobalProxTree.predict did not return a 'prediction' column.")
         # Depending on severity, you might want to raise an error here
         # raise ValueError("Prediction column missing from GlobalProxTree output.")
         # For now, we'll let it proceed, and the Evaluator will likely fail gracefully.
    
    logger.info("Finished external prediction with GlobalProxTree.")

    return predictions_df


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
        self.logger.setLevel(logging.ERROR)
        
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

    def generate_predictions_local(self, test_df: DataFrame) -> DataFrame:
        
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
    
    