# local_model_manager.py
"""
This file is in charge of training our local models.
It takes the preprocessed Spark DataFrame and splits it into parts.
For each part, it trains a Proximity Tree model.
Then, it gathers all the models into one Proximity Forest ensemble that we can use later to make predictions.

NOTE: Trains models in parallel across Spark worker nodes. Number of trees = number of partitions.
"""

import pickle
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame, Window
from aeon.classification.distance_based import ProximityTree, ProximityForest
from pyspark.sql import functions as F
import logging

class LocalModelManager:
    """
    This class handles training local models (Proximity Trees) on chunks of our data and then
    puts them together into an  Proximity Forest ensemble.
    
    The steps are pretty simple:
      1. Get a preprocessed Spark DataFrame.
      2. Split it into parts.
      3. Train a Proximity Tree  model on each part .
      4. Then, it gathers all the trees into one Proximity Forest ensemble
    """

    def __init__(self, config: dict):
        """
        Init with our settings.
        
        Args:
            config (dict): Settings like:
              - num_partitions: How many parts to split the data into.
              - tree_params: Extra parameters for the Proximity Tree  model.
        """       
        # Set default configuration
        self.config = config
        
        # List to store trained trees
        self.trees = []
        
        # Final ensemble model
        self.ensemble = None
        
        # Set up a logger so we can see whats going on
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        

    # def _repartition_data_NotBalanced(self, df: DataFrame) -> DataFrame:
    #     if "num_partitions" in self.config:
    #         new_parts = self.config["num_partitions"]  # ✅ Get value first
    #         self.logger.info(f"Repartitioning data to {new_parts} parts")
    #         return df.repartition(new_parts)
    #     return df
    
    # def _repartition_data_Balanced(self, df: DataFrame, preserve_partition_id: bool = False) -> DataFrame:
    #     if "num_partitions" in self.config and "label_col" in self.config:
    #         num_parts = self.config["num_partitions"]
    #         label_col = self.config["label_col"]
    #         self.logger.info(f"Stratified repartitioning into {num_parts} partitions")
            
    #         # Assign partition IDs (0 to num_parts-1 per class)
    #         # Subtracting 1 so that modulo is computed from 0
    #         window = Window.partitionBy(label_col).orderBy(F.rand())
    #         df = df.withColumn("_partition_id", ((F.row_number().over(window) - 1) % num_parts).cast("int"))
            
    #         # Force exact number of partitions using partition_id
    #         df = df.repartition(num_parts, F.col("_partition_id"))
            
    #         # For production, drop the helper column.
    #         if not preserve_partition_id:
    #             df = df.drop("_partition_id")
    #         return df
    #     return df
        
        
    def _set_forest_classes(self):
        """Collect all class labels from individual trees and mark the forest as fitted."""
        all_classes = []
        for tree in self.trees:
            if hasattr(tree, "classes_"):
                all_classes.extend(tree.classes_)

        unique_classes = np.unique(all_classes)
        self.ensemble.classes_ = unique_classes
        self.ensemble.n_classes_ = len(unique_classes)

        # AEON’s BaseClassifier typically expects a '_class_dictionary' mapping class->int
        self.ensemble._class_dictionary = {
            cls: idx for idx, cls in enumerate(unique_classes)
        }

        # Some older AEON versions store the number of classes in a private attribute
        self.ensemble._n_classes = len(unique_classes)

        # If n_jobs is used, set it explicitly here
        if "n_jobs" in self.config["forest_params"]:
            self.ensemble._n_jobs = self.config["forest_params"]["n_jobs"]

        # BaseClassifier sets 'is_fitted = True' at the end of fit().
        # So we must set the public property 'is_fitted' (not just 'is_fitted_').
        # This ensures ._check_is_fitted() passes in predict().
        self.ensemble.is_fitted_ = True
        self.ensemble.is_fitted = True

        
    def get_ensemble(self) -> ProximityForest:
        """
        Return the trained Proximity Forest ensemble.
        """
        return self.ensemble

    def print_ensemble_details(self):
        """
        Print the details of the aggregated Proximity Forest ensemble.
        """
        if self.ensemble and hasattr(self.ensemble, 'trees_'):
            num_trees = len(self.ensemble.trees_)
            print(f"Aggregated Proximity Forest (contains {num_trees} trees):")
            print(f"  Number of trees (in trees_ attribute): {num_trees}")
            # You might want to print a summary of the parameters used for the forest here
            print(f"  Forest Parameters: {self.ensemble.get_params()}")
            for i, tree in enumerate(self.ensemble.trees_):
                print(f"  Tree {i+1} Details:")
                self._print_tree_node_info(tree.root, depth=2)
            print("-" * 20)
        else:
            print("Proximity Forest ensemble has not been trained yet or the 'trees_' attribute is missing.")

    def _print_tree_node_info(self, node, depth):
        indent = "  " * depth
        print(f"{indent}Node ID: {node.node_id}, Leaf: {node._is_leaf}")

        if node._is_leaf:
            print(f"{indent}  Label: {node.label}, Class Distribution: {node.class_distribution}")
        else:
            splitter = node.splitter
            if splitter:
                exemplars = splitter[0]
                distance_info = splitter[1]
                distance_measure = list(distance_info.keys())[0]
                distance_params = distance_info[distance_measure]

                print(f"{indent}  Splitter:")
                print(f"{indent}    Distance Measure: {distance_measure}, Parameters: {distance_params}")
                print(f"{indent}    Exemplar Classes: {list(exemplars.keys())}")

                print(f"{indent}  Children:")
                for label, child_node in node.children.items():
                    print(f"{indent}    Branch on exemplar of class '{label}':")
                    self._print_tree_node_info(child_node, depth + 1)

   
   
    def train_ensemble(self, df: DataFrame) -> ProximityForest:
        
        """
             Train a forest model iin 3 steps:
        1. Prepare data partitions
        2. Train trees on each partition
        3. Combine trees into a forest
        
        """
        # Repartition the data if our config says so
        #df = self._repartition_data_Balanced(df) - already partitioned!
        
        tree_params = self.config["tree_params"]      
        
         # Define how to process each partition - inline function
        def process_partition(partition_data):
            """Process one data partition to train a tree."""
            try:
                # Convert Spark rows to pandas DataFrame
                pandas_df = pd.DataFrame([row.asDict() for row in partition_data])
                if pandas_df.empty:
                    return []
                
                # Prepare features (3D format for AEON) and labes
                X = np.ascontiguousarray(pandas_df.drop("label", axis=1).values)
                X_3d = X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, 1, features)
                y = pandas_df["label"].values
                
                # Train one tree
                tree = ProximityTree(**tree_params)
                tree.fit(X_3d, y)
                
                # Return serialized tree
                return [pickle.dumps(tree)]
            
            except Exception as e:
                print(f"Failed to train tree on partition: {str(e)}")
                return []  # Skip failed partitions
            
        # Run training on all partitions
        trees_rdd = df.rdd.mapPartitions(process_partition)
        serialized_trees = trees_rdd.collect()
        self.trees = [pickle.loads(b) for b in serialized_trees if b is not None]

        # Build the forest
        if self.trees:
            self.ensemble = ProximityForest(
                n_trees=len(self.trees),
                **self.config["forest_params"]
            )
            # Manually set forest properties
            self.ensemble.trees_ = self.trees
            self._set_forest_classes() 
            return self.ensemble
        else:
            print("Warning: No trees were trained!")
            return None
    
# ==================================================== TESTING ====================================================
# This is a test script to validate the functionality of the LocalModelManager class.
# It creates a dummy dataset, initializes the LocalModelManager, and tests the stratified repartitioning and training of the ensemble.
# It also prints the repartitioned DataFrame and the distribution of labels per partition.
# It is not part of the LocalModelManager class and should be run separately.   


# from pyspark.sql import SparkSession


# spark = SparkSession.builder \
#     .appName("StratifiedRepartitionTest") \
#     .master("local[*]") \
#     .getOrCreate()

# # Create dummy dataset
# data = (
#     [(10, [5.0, 6.0]) for _ in range(4)] +   # Class 0: 5 samples
#     [(11, [6.0, 7.0]) for _ in range(11)] +   # Class 1: 8 samples
#     [(12, [7.0, 8.0]) for _ in range(3)]     # Class 2: 3 samples
# )
# df = spark.createDataFrame(data, ["label", "features"])
# print("Original DataFrame:")
# df.show(15, truncate=False)

# def test_stratified_repartition():

#     config = {
#         "num_partitions": 3,  # The number of partitions to use
#         "label_col": "label"  # The column with the class labels
#     }
#     processor = LocalModelManager(config)
    
#     # Repartition the data *and* preserve the helper column for verification
#     repartitioned_df = processor._repartition_data_Balanced(df, preserve_partition_id=True)
    
#     # Print the repartitioned DataFrame (with _partition_id) for visual inspection
#     print("Repartitioned DataFrame (with preserved _partition_id):")
#     repartitioned_df.show(truncate=False)
    
#     # =================================================================
#     # Validate Partition Count using our computed key
#     # =================================================================
#     # Group by our computed _partition_id and class label
#     distribution_df = repartitioned_df.groupBy("_partition_id", "label") \
#                                       .count() \
#                                       .orderBy("_partition_id", "label")
#     print("Distribution of labels per computed _partition_id:")
#     distribution_df.show(truncate=False)
    
#     # Calculate total counts per label 
#     total_counts_df = repartitioned_df.groupBy("label").count().orderBy("label")
#     print("Total counts per label:")
#     total_counts_df.show(truncate=False)
      
#     print("All tests passed successfully.")


# # Run the test
# test_stratified_repartition()
