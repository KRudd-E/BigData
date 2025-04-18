import pickle
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from aeon.classification.distance_based import ProximityTree, ProximityForest
import logging
from random import sample
from dtaidistance import dtw

class GlobalModelManager:
    def __init__(self):
        self.num_exemplars = 3
        self.num_partitions = 2

    def train(self, df: DataFrame) -> ProximityForest:
        rdd = self.repartition_data(df)

        choose_exemplars = self.choose_exemplars_function(self.num_exemplars)
        rdd_with_exemplar_column = rdd.mapPartitions(choose_exemplars)

        rdd_with_dtw = rdd_with_exemplar_column.mapPartitions(self.calc_dtw_distance)

        rdd_with_closest_exemplar = rdd_with_dtw.mapPartitions(self.assign_closest_exemplar)

        rdd_with_gini = rdd_with_closest_exemplar.mapPartitions(self.calculate_partition_gini) # gini impurity before splitting

        rdd_splits = rdd_with_closest_exemplar.mapPartitions(self.evaluate_splits_within_partition)

        return rdd_splits.collect()

    def partition_data(self, df: DataFrame) -> DataFrame:
        rdd = df.rdd
        repartitioned_rdd = rdd.repartition(self.num_partitions)
        return repartitioned_rdd

    def choose_exemplars_function(self, num_exemplars):
        def choose_exemplars(iterator):
            partition_data = list(iterator)
            exemplars = []
            for row in sample(partition_data, min(num_exemplars, len(partition_data))):
                exemplars.append(row['time_series'])
            return iter([{**row, "exemplars": exemplars} for row in partition_data])
        return choose_exemplars
    
    def calc_dtw_distance(self, iterator):
        partition_data = list(iterator)
        updated_rows = []
        
        for row in partition_data:
            time_series = row['time_series']
            exemplars = row['exemplars']
            
            dtw_distances = [dtw.distance(time_series, exemplar) for exemplar in exemplars]
            
            updated_row = {**row}

            for i, dtw_distance in enumerate(dtw_distances):
                updated_row[f"dtw_distance_exemplar_{i+1}"] = dtw_distance
            
            updated_rows.append(updated_row)
        
        return iter(updated_rows)
    
    def assign_closest_exemplar(self, iterator):
        partition_data = list(iterator)

        for row in partition_data:
            # Check if there are DTW distances for exemplars
            exemplar_distances = {key: value for key, value in row.items() if key.startswith("dtw_distance_exemplar_")}
            
            if exemplar_distances:
                # Find the exemplar with the smallest DTW distance
                closest_exemplar = min(exemplar_distances, key=exemplar_distances.get)
                
                # Assign the closest exemplar to the row
                row["closest exemplar"] = closest_exemplar

        return iter(partition_data)
    
    def calculate_partition_gini(self, iterator):
        partition_data = list(iterator)
        labels = [row['label'] for row in partition_data]

        # Calculate Gini impurity for the partition
        label_counts_dict = {}
        for label in labels:
            label_counts_dict[label] = label_counts_dict.get(label, 0) + 1

        total = sum(label_counts_dict.values())
        proportion_sqrd_values = [(count / total) ** 2 for count in label_counts_dict.values()]
        gini_impurity = 1 - sum(proportion_sqrd_values)

        # Add Gini impurity to each row in the partition
        updated_rows = []
        for row in partition_data:
            updated_row = {**row, "partition_gini": gini_impurity}
            updated_rows.append(updated_row)

        return iter(updated_rows)
    
    def calculate_gini(self, labels):
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        total = sum(label_counts.values())
        gini = 1 - sum((count / total) ** 2 for count in label_counts.values())
        return gini

    def evaluate_splits_within_partition(self, iterator):
        partition_data = list(iterator)
        
        # If the partition is empty, return an empty iterator
        if not partition_data:
            return iter([])
        
        # Get all unique exemplar names in the partition
        unique_exemplars = set(row['closest_exemplar'] for row in partition_data)
        
        results = []
        
        # Loop through each exemplar to evaluate splits
        for exemplar_name in unique_exemplars:
            # Split the data based on the current exemplar
            yes_split = [row for row in partition_data if row['closest_exemplar'] == exemplar_name]
            no_split = [row for row in partition_data if row['closest_exemplar'] != exemplar_name]
            
            # Calculate metrics for the split (e.g., Gini impurity)
            yes_labels = [row['label'] for row in yes_split]
            no_labels = [row['label'] for row in no_split]
            
            yes_gini = self.calculate_gini(yes_labels)
            no_gini = self.calculate_gini(no_labels)
            
            # Store the results for this split
            results.append({
                'exemplar': exemplar_name,
                'yes_gini': yes_gini,
                'no_gini': no_gini,
                'yes_split_size': len(yes_split),
                'no_split_size': len(no_split)
            })
        
        # Return the results as an iterator
        return iter(results)