# config.py
"""
config.py

This file holds configuration settings and constants.
It stores paths, hyperparameters, and Spark settings in one place,
so they can be easily managed and updated as the project grows.
Typically, it's created early on, but it can be refined later.
"""


config = {
    "databricks_data_path": "/mnt/2025-team6/fulldataset_ECG5000.csv",
    "local_data_path": "/fulldataset_ECG5000.csv",
    "label_col": "label",
    "data_percentage": 1.0,
    "min_number_iterarations": 2,

    "local_model_config": {
        "test_local_model" : True,
        "num_partitions": 10,  
        "tree_params": {
            "n_splitters": 5,  # Matches ProximityTree default
            "max_depth": None,  
            "min_samples_split": 5,  # From ProximityTree default
            "random_state": 123
            },
        "forest_params": {
            "random_state": 123,
            "n_jobs": -1  # Use all available cores
            }
    },
    "global_model_config": {
        "test_local_model" : False,
        "num_partitions": 10
    }
}