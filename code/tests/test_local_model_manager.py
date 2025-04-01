import pytest
from pyspark.sql import SparkSession
import pandas as pd
from aeon.classification.distance_based import ProximityForest
import sys
import os

# Add the project root to Python path so that src is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.local_model_manager import LocalModelManager  # assuming your file is named local_model_manager.py

# Fixture for Spark session
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("pytest-local-model-manager") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture for sample Spark DataFrame
@pytest.fixture
def sample_data(spark_session):
    # Create a small Spark DataFrame with a label and three feature columns.
    # Each sample now has 3 points, which is required by ProximityForest.
    data = [
        (1, 10.0, 20.0, 30.0),
        (0, 20.0, 30.0, 40.0),
        (1, 30.0, 40.0, 50.0),
        (0, 40.0, 50.0, 60.0)
    ]
    df = spark_session.createDataFrame(data, ["label", "feature1", "feature2", "feature3"])
    return df

# Fixture for LocalModelManager instance
@pytest.fixture
def model_manager():
    # Config with 2 partitions and model parameters (e.g., setting random_state for reproducibility)
    config = {
        "num_partitions": 2,
        "model_params": {"random_state": 42}
    }
    return LocalModelManager(config)

def test_repartition_data(spark_session, sample_data, model_manager):
    """
    Test that _repartition_data changes the number of partitions based on our config.
    """
    original_parts = sample_data.rdd.getNumPartitions()
    df_repart = model_manager._repartition_data(sample_data)
    new_parts = df_repart.rdd.getNumPartitions()
    # We expect the new partition count to be what we set in the config (2)
    assert new_parts == 2, f"Expected 2 partitions but got {new_parts}"

def test_get_partition_data(spark_session, sample_data, model_manager):
    """
    Test that _get_partition_data returns a pandas DataFrame for each partition.
    """
    df_repart = sample_data.repartition(2)
    for partition_id in range(2):
        pandas_df = model_manager._get_partition_data(df_repart, partition_id)
        # If there's data, we should get a pandas DataFrame
        if pandas_df is not None:
            assert isinstance(pandas_df, pd.DataFrame), "Output is not a pandas DataFrame"
        else:
            # It's fine if a partition is empty
            assert pandas_df is None

def test_train_partition_model(model_manager):
    """
    Test that _train_partition_model trains a ProximityForest
    on a given pandas DataFrame.
    """
    # Create a small pandas DataFrame with a label and three features.
    data = {
        "label": [0, 1, 0, 1],
        "feature1": [10.0, 20.0, 30.0, 40.0],
        "feature2": [100.0, 200.0, 300.0, 400.0],
        "feature3": [1000.0, 2000.0, 3000.0, 4000.0]
    }
    pdf = pd.DataFrame(data)
    model = model_manager._train_partition_model(pdf)
    # Check that the returned model is a ProximityForest
    assert isinstance(model, ProximityForest), "Model is not a ProximityForest"
    # We assume a fitted ProximityForest model has an attribute 'trees_' (adjust if needed)
    assert hasattr(model, "trees_"), "Model does not seem to be fitted"

def test_train_ensemble(spark_session, sample_data, model_manager):
    """
    Test that train_ensemble returns an ensemble of models (one per partition with data)
    and that each model is a ProximityForest.
    """
    df_repart = sample_data.repartition(2)
    models = model_manager.train_ensemble(df_repart)
    # Expect at least one model in the ensemble
    assert len(models) > 0, "No models were trained"
    # Check that each model is a ProximityForest
    for m in models:
        assert isinstance(m, ProximityForest), "One of the ensemble models is not a ProximityForest"

# Run tests if this script is executed directly.
if __name__ == "__main__":
    pytest.main([__file__])
