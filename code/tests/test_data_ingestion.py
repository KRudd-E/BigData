import pytest
from pyspark.sql import SparkSession
import sys
import os

# Add the project root to Python path so that src is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import DataIngestion  

@pytest.fixture(scope="session")
def spark_session():
    # Create a local Spark session for testing.
    spark = SparkSession.builder \
        .appName("pytest-pyspark") \
        .master("local[*]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_config(tmp_path):
    # Create a temporary TSV file with sample data
    # Mimicking the data: three columns, no header.
    data = "1\tAlice\t25\n2\tBob\t30\n"
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text(data)
    # Return the config with the file path (absolute path)
    return {"data_path": str(tsv_file)}

def test_load_data(spark_session, sample_config):
    """Test that load_data returns a DataFrame with the expected row and column counts."""
    ingestion = DataIngestion(spark_session, sample_config)
    df = ingestion.load_data()
    # Expect 2 rows and 3 columns as per our sample data.
    assert df.count() == 2
    assert len(df.columns) == 3
    # Log the schema and sample data
    print("Data schema:")
    df.printSchema()
    print("Showing sample data:")

def test_validate_data(spark_session, sample_config):
    """Test that validate_data raises an exception when the DataFrame is empty."""
    ingestion = DataIngestion(spark_session, sample_config)
    # Create an empty DataFrame with a simple schema.
    empty_df = spark_session.createDataFrame([], "a string, b string, c int")
    with pytest.raises(Exception, match="Data is empty!"):
        ingestion.validate_data(empty_df)

def test_get_sample_data(spark_session, sample_config):
    """Test that get_sample_data returns a sample DataFrame with fewer or equal rows."""
    ingestion = DataIngestion(spark_session, sample_config)
    # Create a DataFrame with three rows.
    data = [("1", "Alice", 25), ("2", "Bob", 30), ("3", "Charlie", 35)]
    df = spark_session.createDataFrame(data, ["a", "b", "c"])
    sample_df = ingestion.get_sample_data(df, fraction=0.5)
    # The sampled DataFrame should have less than or equal to 3 rows.
    assert sample_df.count() <= 3

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
