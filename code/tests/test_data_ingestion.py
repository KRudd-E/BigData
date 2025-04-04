import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import sys
import os

# Add the project root to Python path so that src is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import DataIngestion

@pytest.fixture(scope="session")
def spark_session():
    """Create a local Spark session for testing."""
    spark = SparkSession.builder \
        .appName("pytest-pyspark") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary csv file with sample data and return its path in the config."""
    header = "label," + ",".join([f"_c{i}" for i in range(1, 141)]) + "\n"
    data = "1," + ",".join(["0.0"] * 140) + "\n"
    data += "0," + ",".join(["1.0"] * 140) + "\n"
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(header + data)
    return {"data_path": str(csv_file)}

@pytest.fixture
def empty_config(tmp_path):
    """Create a temporary empty csv file and return its path in the config."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("label," + ",".join([f"_c{i}" for i in range(1, 141)]) + "\n")
    return {"data_path": str(csv_file)}

@pytest.fixture
def default_config():
    """Return a config with the default data path."""
    return {}

def test_load_data(spark_session, sample_config):
    """Test that load_data returns a DataFrame with the expected schema and row count."""
    ingestion = DataIngestion(spark_session, sample_config)
    df = ingestion.load_data()

    # Expect 2 rows based on the sample data
    assert df.count() == 2

    # Expect 'label' and 140 feature columns
    expected_columns = ["label"] + [f"_c{i}" for i in range(1, 141)]
    assert sorted(df.columns) == sorted(expected_columns)

    # Expect the correct data types
    assert isinstance(df.schema["label"].dataType, IntegerType)
    for i in range(1, 141):
        assert isinstance(df.schema[f"_c{i}"].dataType, DoubleType)

    print("\nData schema:")
    df.printSchema()
    print("Showing sample data:")
    df.show()

def test_validate_data(spark_session, sample_config):
    """Test that validate_data raises an exception when the DataFrame is empty."""
    ingestion = DataIngestion(spark_session, sample_config)
    # Create an empty DataFrame with the expected schema
    schema = StructType([
        StructField("label", IntegerType(), True),
        *[StructField(f"_c{i}", DoubleType(), True) for i in range(1, 141)]
    ])
    empty_df = spark_session.createDataFrame([], schema)
    with pytest.raises(Exception, match="Data is empty!"):
        ingestion.validate_data(empty_df)

def test_load_data_with_sampling(spark_session, sample_config):
    """Test that load_data correctly applies data sampling."""
    config_with_percentage = sample_config.copy()
    config_with_percentage["data_percentage"] = 0.5  # 50% sampling (as a float)
    ingestion = DataIngestion(spark_session, config_with_percentage)
    df = ingestion.load_data()

    assert 0 < df.count() <= 2
    assert len(df.columns) == 141

def test_validate_data_not_empty(spark_session, sample_config):
    """Test that validate_data does not raise an exception for a non-empty DataFrame."""
    ingestion = DataIngestion(spark_session, sample_config)
    df = ingestion.load_data()
    try:
        ingestion.validate_data(df)
    except Exception as e:
        pytest.fail(f"validate_data raised an exception for non-empty DataFrame: {e}")

def test_validate_data_empty(spark_session, empty_config):
    """Test that validate_data raises an exception when the DataFrame is empty."""
    ingestion = DataIngestion(spark_session, empty_config)
    with pytest.raises(Exception, match="Data is empty!"):
        ingestion.load_data()

def test_get_sample_data(spark_session, sample_config):
    """Test that get_sample_data returns a sample DataFrame with the specified fraction."""
    ingestion = DataIngestion(spark_session, sample_config)
    df = ingestion.load_data()
    fraction = 0.5  # 50% sampling (as a float)
    sample_df = ingestion.get_sample_data(df, fraction=fraction)
    # The sampled DataFrame should have approximately fraction * total rows
    assert sample_df.count() <= df.count()
    assert sorted(sample_df.columns) == sorted(df.columns)

def test_get_sample_data_default_fraction(spark_session, sample_config):
    """Test that get_sample_data returns a sample DataFrame with the default fraction."""
    ingestion = DataIngestion(spark_session, sample_config)
    df = ingestion.load_data()
    sample_df = ingestion.get_sample_data(df)
    assert sample_df.count() <= df.count()
    assert sorted(sample_df.columns) == sorted(df.columns)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
