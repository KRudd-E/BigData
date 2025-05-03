# controller.py
"""
controller.py

This file ties the whole pipeline together.
It iterates through different partition configurations, running the 
full pipeline (ingestion, preprocessing, training, prediction, evaluation) 
in each iteration to assess the impact of partitioning.
"""

import os
import json
import numpy as np
from pyspark.sql import SparkSession, DataFrame # Added DataFrame import
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from local_model_manager import LocalModelManager
from global_model_manager import GlobalModelManager
from prediction_manager import PredictionManager, predict_with_global_prox_tree
from evaluation import Evaluator
from utilities import show_compact, randomSplit_dist, randomSplit_stratified_via_sampleBy, compute_min_max
from visualization import plot_confusion_matrix, plot_class_metrics
import logging
import time
import pickle 
import pathlib 

class PipelineController_Loop:
    def __init__(self, config):
        """
        Initialize the controller with the pipeline configuration.
        """
        self.config = config
        self.spark = None
        self.ingestion_config = {}
        # Logger setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            if self.logger.level == logging.NOTSET:
                 self.logger.setLevel(logging.INFO) 
            self.logger.propagate = False

    def _setup_spark(self):
        """
        Setup or retrieve the Spark session based on the environment.
        Handles stopping existing local sessions for clean iteration state.
        Constructs appropriate data paths.
        Adds necessary Python files to Spark context for local runs.
        """
        # Stop existing local session if present
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
             existing_spark = SparkSession.getActiveSession()
             if existing_spark:
                  self.logger.info("Stopping existing Spark session before starting new one.")
                  existing_spark.stop()

        # Configure Spark Session
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            # Databricks environment
            self.spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            print("\nUsing Databricks Spark session.")
            self.ingestion_config = {
                "data_path": self.config.get("databricks_data_path", "/mnt/2025-team6/fulldataset_ECG5000.csv"),
                "data_percentage": self.config.get("data_percentage", 0.05) 
            }
        else:
            # Local environment setup
            self.spark = SparkSession.builder \
                .appName("LocalPipeline") \
                .master("local[6]") \
                .config("spark.driver.memory", "12g") \
                .config("spark.executor.memory", "12g") \
                .config("spark.driver.maxResultSize", "12g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
            print("\nUsing local Spark session.")

            current_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.ingestion_config = {
                "data_path": project_root + self.config.get("local_data_path", "/fulldataset_ECG5000.csv"),
                "data_percentage": self.config.get("data_percentage", 1.0)
            }
            self.logger.info(f"Project root estimated as: {project_root}")
            self.logger.info(f"Data path set to: {self.ingestion_config['data_path']}")
        

        # Add Python module dependencies for local runs
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ and self.spark:
            modules_to_add = ['global_model_manager.py'] # Add other required modules if needed
            try:
                current_dir = pathlib.Path(os.path.abspath(__file__)).parent
                for module_name in modules_to_add:
                     module_path = current_dir / module_name
                     if module_path.exists():
                         self.spark.sparkContext.addPyFile(str(module_path)) # addPyFile needs string path
                         self.logger.debug(f"Added {module_path} to SparkContext pyFiles.") 
                     else:
                          self.logger.error(f"Could not find module {module_name} at {module_path}")
            except NameError:
                 self.logger.warning("__file__ not defined, cannot automatically add Python modules.")
            except Exception as e:
                self.logger.error(f"Failed to add Python module to SparkContext: {e}")
            
    def run(self):
        """
        Executes the main pipeline loop for model training and evaluation.
        Iterates through specified partition counts, running the full pipeline 
        (ingestion, preprocessing, training, eval, save) in each iteration.
        """
        # --- Determine number of iterations and which models to run ---
        number_iterations_global, number_iterations_local = 0, 0
        run_local = self.config.get("local_model_config", {}).get("test_local_model", False)
        run_global = self.config.get("global_model_config", {}).get("test_global_model", False)

        if not run_local and not run_global:
            self.logger.error("No model selected for testing. Set 'test_local_model' or 'test_global_model' to True in config.")
            return 

        if run_local:
            number_iterations_local = self.config.get("local_model_config", {}).get("num_partitions", 0)
        if run_global:
            # Use the partition count from global config to control loop iterations for the experiment
            number_iterations_global = self.config.get("global_model_config", {}).get("num_partitions", 0) 
            
        # Determine the overall maximum number of iterations needed
        number_iterations = 0
        if run_local: number_iterations = max(number_iterations, number_iterations_local)
        if run_global: number_iterations = max(number_iterations, number_iterations_global)

        min_iterations = self.config.get("min_number_iterarations", 2)
        start_iteration = min_iterations
        end_iteration = number_iterations
        
        if end_iteration < start_iteration:
             self.logger.warning(f"Max iterations ({end_iteration}) is less than min iterations ({start_iteration}). No iterations will run.")
             start_iteration = end_iteration + 1 # Make range empty

        # --- Initialize report accumulators ---
        all_reports_global = {} 
        all_reports_local = {}  

        self._setup_spark() 
        if not self._setup_spark():
             self.logger.error("Initial Spark setup failed. Aborting run.")
             return
        if self.ingestion_config.get("data_path") is None:
             self.logger.error("Initial data path construction failed. Aborting run.")
             if "DATABRICKS_RUNTIME_VERSION" not in os.environ and self.spark: self.spark.stop()
             return
        # ============================================================
        # === Main Iteration Loop ===
        # ============================================================
        for i in range(start_iteration, end_iteration + 1): 
            self.logger.info(f"========== Starting Iteration {i} ==========")
            
            iteration_run_local = run_local and i <= number_iterations_local 
            iteration_run_global = run_global # Global runs in all iterations up to its max if enabled

            current_datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
            # Initialize modules for this iteration
            self.evaluator = Evaluator(track_memory=self.config.get("track_memory", False)) 
            # Pass the iteration-specific config to Preprocessor
            current_iter_config = self.config.copy()
            if iteration_run_local: current_iter_config["local_model_config"]["num_partitions"] = i
            if iteration_run_global: current_iter_config["global_model_config"]["num_partitions"] = i
            self.preprocessor = Preprocessor(config=current_iter_config) 
            self.ingestion = DataIngestion(spark=self.spark, config=self.ingestion_config)

            preprocessed_train_df: DataFrame = None
            preprocessed_test_df: DataFrame = None

            try:
                # --- Data Ingestion ---
                self.evaluator.start_timer("Ingestion")
                self.logger.info(f"Iteration {i}: Loading data...")
                df = self.ingestion.load_data() 
                if df.limit(1).count() == 0:
                     self.logger.error(f"Iteration {i}: Data ingestion resulted in empty DataFrame. Skipping iteration.")
                     if "DATABRICKS_RUNTIME_VERSION" not in os.environ: self.spark.stop()
                     continue 
                self.evaluator.record_time("Ingestion")
                self.logger.info(f"Iteration {i}: Initial DataFrame loaded with {df.count()} rows.") 

                # --- Split, Calculate Min/Max ---
                self.evaluator.start_timer("Split_MinMax")
                min_max_values = compute_min_max(df) 
                self.logger.info(f"Iteration {i}: Computed Min-Max values.") 
                train_df, test_df = randomSplit_stratified_via_sampleBy(df, label_col="label", weights=[0.8, 0.2], seed=123)       
                self.evaluator.record_time("Split_MinMax")
                
                # --- Preprocessing Train Data (Includes Repartitioning based on 'i' via Preprocessor config) ---
                self.evaluator.start_timer("Preprocessing_Train")
                self.logger.info(f"Iteration {i}: Preprocessing train data (partitions={i})...")
                preprocessed_train_df = self.preprocessor.run_preprocessing(train_df, min_max_values) 
                train_count = preprocessed_train_df.count() # Action to materialize preprocessing
                self.logger.info(f"Iteration {i}: Preprocessed train data count: {train_count}")
                self.evaluator.record_time("Preprocessing_Train")

                # --- Preprocessing Test Data (Uses same partition logic for consistency, though maybe not needed) ---
                self.evaluator.start_timer("Preprocessing_Test")
                self.logger.info(f"Iteration {i}: Preprocessing test data (partitions={i})...")
                preprocessed_test_df = self.preprocessor.run_preprocessing(test_df, min_max_values)
                test_count = preprocessed_test_df.count() # Action
                self.logger.info(f"Iteration {i}: Preprocessed test data count: {test_count}")
                self.evaluator.record_time("Preprocessing_Test")

                if train_count == 0 or test_count == 0:
                    self.logger.error(f"Iteration {i}: Preprocessing resulted in empty train or test set. Skipping model steps.")
                    continue 

            except Exception as e:
                 if "Path does not exist" in str(e):
                      spark_path_attempt = self.ingestion_config.get("data_path", "N/A") 
                      self.logger.error(f"Iteration {i}: Spark failed to find data file during load. Path: {spark_path_attempt}. Error: {e}", exc_info=False) 
                 elif isinstance(e, (ConnectionRefusedError, ConnectionResetError)) or "Connection reset by peer" in str(e):
                      self.logger.error(f"Iteration {i}: Connection error during data processing: {e}", exc_info=True)
                 else:
                      self.logger.error(f"Iteration {i}: Error during data processing: {e}", exc_info=True) 
                 if "DATABRICKS_RUNTIME_VERSION" not in os.environ: self.spark.stop()
                 continue # Skip to next iteration

            # Define variables to hold results for the iteration
            model_ensamble = None 
            predictions_df = None 
            
            # =================== GLOBAL MODEL ===================
            if iteration_run_global:
                self.global_model_manager = None 
                model_ensamble = None 
                global_report = None 
                try:
 
                    global_config = current_iter_config.get("global_model_config", {}) 
                    if not global_config:
                         self.logger.error(f"Iteration {i}: global_model_config missing in current_iter_config. Skipping global model.")
                         continue # Skip if config section is missing
                    # Pass the config containing the num_partitions for this iteration
                    self.global_model_manager = GlobalModelManager(spark=self.spark, config=global_config) 
                    
                    print(f"\nIteration (Partitions:) {i}: Train global model......")
                    self.evaluator.start_timer("Global_Training")
                    # Train on the data preprocessed (and potentially repartitioned) for this iteration
                    model_ensamble = self.global_model_manager.fit(preprocessed_train_df) 
                    self.evaluator.record_time("Global_Training")
                    print(f"Iteration {i}: Finish Global Training.")

                    # Check if training was successful
                    if model_ensamble and hasattr(model_ensamble, 'tree') and len(model_ensamble.tree) > 1:
                        self.logger.info(f"Iteration {i}: Global model training successful.")
                        
                        # Prediction
                        print(f"\nIteration {i}: Generate predictions with global model......")
                        self.evaluator.start_timer("Global_Prediction")
                        predictions_df = predict_with_global_prox_tree(model_ensamble, preprocessed_test_df) 
                        self.evaluator.record_time("Global_Prediction")
                        print(f"Iteration {i}: Finish Global Prediction.")
                        
                        # print(f"\nIteration {i}: Global Model Predictions Distribution:")
                        # predictions_df.groupBy("prediction").count().show() 

                        # Evaluation
                        print(f"\nIteration {i}: Generate metrics for global model......")
                        global_report, class_names = self.evaluator.log_metrics(predictions_df, model=model_ensamble) 
                        all_reports_global[str(i)] = global_report 
                        print(f"Iteration {i}: Finish Global Evaluation.")
                        
                        depth = global_config.get("tree_params", {}).get("max_depth", "NA")
                        
                        # --- Save Model --- 
                        model_folder = "models_global" 
                        os.makedirs(model_folder, exist_ok=True) 
                        model_filename = f"global_model_parti_{i}_{current_datetime}_depth_{depth}.pkl" 
                        model_save_path = os.path.join(model_folder, model_filename)
                        try:
                            self.global_model_manager.save_tree(model_save_path) 
                            print(f"Saved global model to {model_save_path}")
                        except Exception as e: self.logger.error(f"Failed to save global model {model_filename}: {e}")

                    else:
                        self.logger.warning(f"Iteration {i}: Global model training failed or resulted in trivial tree. Skipping prediction/evaluation.")

                except Exception as e:
                    self.logger.error(f"Iteration {i}: Error during global model processing: {e}", exc_info=True) 
                finally:
                    # Clean up global model objects
                    self.global_model_manager = None 
                    if 'model_ensamble' in locals() and model_ensamble is not None: del model_ensamble
                    if 'predictions_df' in locals() and predictions_df is not None: del predictions_df
                    self.logger.debug(f"Iteration {i}: Cleaned up global model objects.")

            # =================== LOCAL MODEL ===================
            if iteration_run_local:
                self.local_model_manager = None 
                model_ensamble = None 
                predictions_df = None 
                local_report = None 
                try:
                    
                    local_config = current_iter_config.get("local_model_config", {})
                    if not local_config:
                        self.logger.error(f"Iteration {i}: local_model_config missing in current_iter_config. Skipping local model.")
                        continue # Skip if config section is missing     

                    # Pass the config containing the num_partitions for this iteration
                    self.local_model_manager = LocalModelManager(config=local_config) 
                    model_ensamble = self.local_model_manager.train_ensemble(preprocessed_train_df) # This is the ProximityForest object
                    self.evaluator.start_timer("Local_Training")
                    print(f"\nIteration (Partitions:) {i}: Train local model with {i} partitions......")
                    # Train on the data preprocessed (and potentially repartitioned) for this iteration
                    
                    self.evaluator.record_time("Local_Training")
                    print(f"Iteration {i}: Finish Local Training.")
                    
                    # Check if training was successful
                    if model_ensamble is not None and hasattr(model_ensamble, 'trees_') and model_ensamble.trees_:
                        self.logger.info(f"Iteration {i}: Local model training successful.")
                        
                        # Prediction
                        print(f"\nIteration {i}: Generate predictions with local model......")
                        self.evaluator.start_timer("Local_Prediction")
                        self.predictor = PredictionManager(self.spark, model_ensamble) 
                        predictions_df = self.predictor.generate_predictions_local(preprocessed_test_df) 
                        self.evaluator.record_time("Local_Prediction")
                        print(f"Iteration {i}: Finish Local Prediction.")
                    
                        # print(f"\nIteration {i}: Local model Predictions Distribution:")
                        # predictions_df.groupBy("prediction").count().show() 

                        # Evaluation
                        print(f"\nIteration {i}: Generate metrics for local model......")
                        local_report, class_names = self.evaluator.log_metrics(predictions_df, model=model_ensamble)
                        all_reports_local[str(i)] = local_report 
                        print(f"Iteration {i}: Finish Local Evaluation.")
                        
                        depth = local_config.get("tree_params", {}).get("max_depth", "NA") 
                        
                        # --- Save Model --- 
                        model_folder = "models_local" 
                        os.makedirs(model_folder, exist_ok=True) 
                        model_filename = f"local_model_parti_{i}_{current_datetime}_depth_{depth}.pkl" 
                        model_save_path = os.path.join(model_folder, model_filename)
                        try:
                            with open(model_save_path, 'wb') as f: pickle.dump(model_ensamble, f) 
                            print(f"Saved local model ensemble to {model_save_path}")
                        except Exception as e: self.logger.error(f"Failed to save local model {model_filename}: {e}")
                        
                    else:
                        self.logger.warning(f"Iteration {i}: Local model training failed or resulted in empty ensemble. Skipping prediction/evaluation.")

                except Exception as e:
                     self.logger.error(f"Iteration {i}: Error during local model processing: {e}", exc_info=True) 
                finally:
                    # Clean up local model objects
                    self.local_model_manager = None 
                    if 'model_ensamble' in locals() and model_ensamble is not None: del model_ensamble
                    if 'predictions_df' in locals() and predictions_df is not None: del predictions_df
                    self.logger.debug(f"Iteration {i}: Cleaned up local model objects.")


            # --- Iteration End ---
            # Cleanup DataFrames for this iteration (optional, Spark manages memory but explicit cleanup can help)
            if 'preprocessed_train_df' in locals() and preprocessed_train_df is not None: del preprocessed_train_df
            if 'preprocessed_test_df' in locals() and preprocessed_test_df is not None: del preprocessed_test_df
            if 'df' in locals() and df is not None: del df # Remove reference to raw df for this iteration
            
            self.logger.info(f"========== Finished Iteration {i} ==========")
            # Optional delay
            if self.config.get('delay_time', 0) > 0 and i < end_iteration : 
                print(f"\nIteration {i}: Waiting for {self.config['delay_time']} seconds before next iteration...\n")
                time.sleep(self.config['delay_time'])

        # ============================================================
        # === Step 3: Save Accumulated Reports After Loop ===
        # ============================================================
        final_datetime = time.strftime("%Y-%m-%d-%H-%M-%S") 
        
        if all_reports_global:
             report_folder = "logs" 
             os.makedirs(report_folder, exist_ok=True) 
             report_filename_global = f"report_global_model_ALL_{final_datetime}.json" 
             report_save_path_global = os.path.join(report_folder, report_filename_global)
             try:
                 with open(report_save_path_global, "w") as f: json.dump(all_reports_global, f, indent=2)
                 print(f"Saved ALL global model reports to {report_save_path_global}") 
             except Exception as e: self.logger.error(f"Failed to save aggregated global report: {e}")

        if all_reports_local:
             report_folder = "logs"
             os.makedirs(report_folder, exist_ok=True) 
             report_filename_local = f"report_local_model_ALL_{final_datetime}.json" 
             report_save_path_local = os.path.join(report_folder, report_filename_local)
             try:
                 with open(report_save_path_local, "w") as f: json.dump(all_reports_local, f, indent=2)
                 print(f"Saved ALL local model reports to {report_save_path_local}") 
             except Exception as e: self.logger.error(f"Failed to save aggregated local report: {e}")


        # --- Pipeline End ---


        # Final Spark stop if running locally (handles case where loop didn't run)
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
             final_spark = SparkSession.getActiveSession()
             if final_spark:
                  final_spark.stop()
                  print(f"\nFinal Spark session stopped (local mode)!")

        print("\n--- Pipeline execution finished ---")

