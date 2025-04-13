from controller import PipelineController



if __name__ == "__main__":
    print("Starting pipeline via controller.py")
    # Example config; adjust paths and parameters as needed.
    config = {
        "databricks_data_path": "/mnt/2025-team6/fulldataset_ECG5000.csv",
        "local_data_path": "/fulldataset_ECG5000.csv",
        "data_percentage": 0.1,
        "local_model_config": {
            "num_partitions": 2, 
            "model_params": {
                "random_state": 1234
                }
            }
    }
    controller = PipelineController(config)
    controller.run()