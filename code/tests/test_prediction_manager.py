import pytest
from unittest.mock import MagicMock
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, DoubleType
from pyspark.sql import functions as F
import numpy as np
from aeon.classification.distance_based import ProximityForest
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction_manager import PredictionManager


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("pytest-prediction_manager")\
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def mock_models():
    mock_model_1 = MagicMock(spec=ProximityForest)
    mock_model_1.predict.return_value = np.array([0])
    mock_model_2 = MagicMock(spec=ProximityForest)
    mock_model_2.predict.return_value = np.array([1])
    return [mock_model_1, mock_model_2]

@pytest.fixture
def prediction_manager(spark, mock_models):
    pm = PredictionManager(spark, mock_models)
    logger = logging.getLogger("prediction_manager")
    log_handler = logging.StreamHandler()
    logger.addHandler(log_handler)
    pm.original_log_level = logger.level
    yield pm
    logger.removeHandler(log_handler)
    logger.setLevel(pm.original_log_level)

def _create_test_dataframe(spark: SparkSession, data: list, schema: StructType) -> DataFrame:
    return spark.createDataFrame(data, schema)

def test_initialization(prediction_manager, spark, mock_models):
    assert prediction_manager.spark == spark
    assert prediction_manager.models == mock_models
    assert isinstance(prediction_manager.logger, logging.Logger)

def test_predict_udf(prediction_manager, spark, mock_models):
    predict_udf = prediction_manager._predict_udf()
    test_data = [(np.array([1.0, 2.0]).tolist(),)]
    schema = StructType([StructField("features", ArrayType(DoubleType()), False)])
    test_df = _create_test_dataframe(spark, test_data, schema)
    result_df = test_df.withColumn("prediction", predict_udf(F.col("features")))
    result = result_df.collect()[0]["prediction"]
    assert result in [0, 1]
    mock_models[0].predict.assert_called_once()
    mock_models[1].predict.assert_called_once()

def test_predict_udf_single_model(spark):
    single_model = [MagicMock(spec=ProximityForest)]
    single_model[0].predict.return_value = np.array([2])
    pm_single_model = PredictionManager(spark, single_model)
    predict_udf = pm_single_model._predict_udf()
    test_data = [(np.array([3.0, 4.0]).tolist(),)]
    schema = StructType([StructField("features", ArrayType(DoubleType()), False)])
    test_df = _create_test_dataframe(spark, test_data, schema)
    result_df = test_df.withColumn("prediction", predict_udf(F.col("features")))
    result = result_df.collect()[0]["prediction"]
    assert result == 2
    single_model[0].predict.assert_called_once()

def test_predict_udf_error_handling(prediction_manager, spark, mock_models, caplog):
    mock_models[0].predict.side_effect = Exception("Prediction error")
    predict_udf = prediction_manager._predict_udf()
    test_data = [(np.array([5.0, 6.0]).tolist(),)]
    schema = StructType([StructField("features", ArrayType(DoubleType()), False)])
    test_df = _create_test_dataframe(spark, test_data, schema)
    with caplog.at_level(logging.ERROR):
        result_df = test_df.withColumn("prediction", predict_udf(F.col("features")))
        result = result_df.collect()[0]["prediction"]
        assert result == -1
        assert "Prediction failed:" in caplog.text
    mock_models[0].predict.assert_called_once()
    mock_models[1].predict.assert_called_once()
    mock_models[0].predict.side_effect = None

def test_generate_predictions(prediction_manager, spark):
    test_schema = StructType([
        StructField("feature1", DoubleType(), False),
        StructField("feature2", DoubleType(), False),
        StructField("label", IntegerType(), False)
    ])
    test_data = [(1.0, 2.0, 0), (3.0, 4.0, 1)]
    test_df = _create_test_dataframe(spark, test_data, test_schema)

    mock_predict_udf = MagicMock(return_value=F.lit(0))
    prediction_manager._predict_udf = mock_predict_udf

    predictions_df = prediction_manager.generate_predictions(test_df)
    assert "prediction" in predictions_df.columns
    assert predictions_df.count() == 2
    assert predictions_df.select("prediction").distinct().collect()[0][0] == 0
    mock_predict_udf.assert_called_once()

def test_generate_predictions_no_models(spark):
    empty_manager = PredictionManager(spark, [])
    test_schema = StructType([
        StructField("feature_a", DoubleType(), False),
        StructField("feature_b", DoubleType(), False)
    ])
    test_data = [(1.0, 2.0)]
    test_df = _create_test_dataframe(spark, test_data, test_schema)
    with pytest.raises(ValueError) as excinfo:
        empty_manager.generate_predictions(test_df)
    assert str(excinfo.value) == "No models available for prediction"

def test_generate_predictions_feature_extraction(prediction_manager, spark):
    test_schema = StructType([
        StructField("col_a", DoubleType(), False),
        StructField("col_b", DoubleType(), False),
        StructField("target", IntegerType(), False),
        StructField("label", IntegerType(), False)
    ])
    test_data = [(1.1, 2.2, 99, 0), (3.3, 4.4, 88, 1)]
    test_df = _create_test_dataframe(spark, test_data, test_schema)

    def mock_predict(features):
        assert len(features) == 2
        assert isinstance(features, list)
        assert all(isinstance(f, float) for f in features)
        return 0

    mock_predict_pyspark_udf = F.udf(mock_predict, IntegerType())
    prediction_manager._predict_udf = MagicMock(return_value=mock_predict_pyspark_udf)

    predictions_df = prediction_manager.generate_predictions(test_df)
    assert "prediction" in predictions_df.columns
    assert "features" not in predictions_df.columns
    prediction_manager._predict_udf.assert_called_once()
    
if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])