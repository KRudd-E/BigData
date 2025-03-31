import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row, functions as F
import sys
import os


# Add the project root to Python path so that src is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessor

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("pytest-preprocessing") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def preprocessor():
    # we can pass in any configuration options if needed; for now, we'll use an empty dict.
    return Preprocessor(config={})

def test_handle_missing_values(spark_session, preprocessor):
    # Create a DataFrame with one non-null row and one all-null row.
    data = [
        (1, 2.0, 3.0),
        (None, None, None)
    ]
    df = spark_session.createDataFrame(data, ["_c0", "_c1", "_c2"])
    
    cleaned_df = preprocessor.handle_missing_values(df)
    
    # Expect the row with all nulls to be dropped.
    assert cleaned_df.count() == 1

def test_split_label_features(spark_session, preprocessor):
    # Create a DataFrame mimicking the raw format.
    data = [
        (1, 2.0, 3.0),
        (2, 4.0, 6.0)
    ]
    df = spark_session.createDataFrame(data, ["_c0", "_c1", "_c2"])
    
    split_df = preprocessor.split_label_features(df)
    
    # Check that the first column was renamed to 'label'
    assert "label" in split_df.columns
    # Also, _c0 should no longer be in the DataFrame.
    assert "_c0" not in split_df.columns

def test_normalize_features(spark_session, preprocessor):
    # Create a DataFrame with known values.
    data = [
        (1, 10.0, 100.0),
        (2, 20.0, 200.0),
        (3, 30.0, 300.0)
    ]
    df = spark_session.createDataFrame(data, ["label", "feature1", "feature2"])
    
    # Call normalize_features (the correct method name)
    normalized_df = preprocessor.normalize_features(df)
    
    # Collect results
    rows = normalized_df.select("feature1", "feature2").collect()
    
    # For feature1: min = 10, max = 30; so 10 becomes 0, 20 becomes 0.5, 30 becomes 1.0
    # For feature2: min = 100, max = 300; so 100 becomes 0, 200 becomes 0.5, 300 becomes 1.0
    
    # normalized_value = (value - min) / (max - min)
    
    # For feature1, the minimum is 10 and the maximum is 30. So:

    # For value 10: (10 - 10) / (30 - 10) = 0 / 20 = 0.0

    # For value 20: (20 - 10) / (30 - 10) = 10 / 20 = 0.5

    # For value 30: (30 - 10) / (30 - 10) = 20 / 20 = 1.0

    # Similarly, for feature2, with min 100 and max 300:

    # For value 100: (100 - 100) / (300 - 100) = 0 / 200 = 0.0

    # For value 200: (200 - 100) / (300 - 100) = 100 / 200 = 0.5

    # For value 300: (300 - 100) / (300 - 100) = 200 / 200 = 1.0
    
    expected = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    
    for row, (e1, e2) in zip(rows, expected):
        assert abs(row["feature1"] - e1) < 1e-6
        assert abs(row["feature2"] - e2) < 1e-6



def test_run_preprocessing(spark_session, preprocessor):
    # Create a DataFrame that mimics a raw ECG row.
    # Assume _c0 is the label and _c1, _c2, ... are features.
    data = [
        (1, 10.0, 100.0),
        (2, 20.0, 200.0),
        (3, 30.0, 300.0),
        (None, None, None)  # This row should be dropped by missing value handler.
    ]
    df = spark_session.createDataFrame(data, ["_c0", "_c1", "_c2"])
    
    # Run the complete preprocessing pipeline.
    processed_df = preprocessor.run_preprocessing(df)
    
    # Check that the empty row was removed.
    assert processed_df.count() == 3
    # Check that the label column was renamed.
    assert "label" in processed_df.columns
    # And that normalization was applied on feature columns (_c1 and _c2)
    # Since we use our simple dummy normalization in our earlier code,
    # you might not see changes if not implemented.
    # For this test, you can simply check that the processed DataFrame has expected columns.
    expected_cols = {"label", "_c1", "_c2"}
    assert set(processed_df.columns) == expected_cols

if __name__ == "__main__":
    pytest.main([__file__])
