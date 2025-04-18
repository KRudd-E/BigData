import pickle
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from aeon.classification.distance_based import ProximityTree, ProximityForest
import logging
from random import sample
from dtaidistance import dtw
from distance_measures import calc_dtw_distance, calc_euclid_distance

class GlobalModelManager:
    def __init__(self):
        self.num_exemplars = 3
        self.num_partitions = 2
        self.distance_types = ['dtw', 'euclidean'] #add more types to this as they are created in distance_measures.py

    def train(self, df: DataFrame) -> ProximityForest:
        rdd = self.partition_data(df) #changed naming of this to match function name 

        choose_exemplars = self.choose_exemplars_function(self.num_exemplars)
        rdd_with_exemplar_column = rdd.mapPartitions(choose_exemplars)

        rdd_with_distance = rdd_with_exemplar_column.mapPartitions(self.calc_distance)

        rdd_with_closest_exemplar = rdd_with_distance.mapPartitions(self.assign_closest_exemplar)

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
    
    def choose_distance_function(self, distance_types):
        samples = np.random.choice(distance_types, size=self.num_partitions, replace=True) 
        return samples
    
    def calc_distance(self, iterator):
        partition_data = list(iterator)
        distance_types = self.choose_distance_function(self.distance_types)

        for row, distance_type in zip(partition_data, distance_types):
            if distance_type == 'dtw':
                row['distance'] = calc_dtw_distance(row['time_series'], row['exemplars'])
            elif distance_type == 'euclidean':
                row['distance'] = calc_euclid_distance(row['time_series'], row['exemplars'])
            # Add more distance types as needed
        return iter(partition_data)
            
           
    def assign_closest_exemplar(self, iterator):
        partition_data = list(iterator)

        for row in partition_data:
            # Check if there are DTW distances for exemplars
            exemplar_distances = {key: value for key, value in row.items() if key.startswith("exemplar_")} #changed to name in distance_measures.py
            
            if exemplar_distances:
                # Find the exemplar with the smallest DTW distance
                closest_exemplar = min(exemplar_distances, key=exemplar_distances.get)
                
                # Assign the closest exemplar to the row
                row["closest_exemplar"] = closest_exemplar #again fixed inconsistent naming

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
    
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("GenericRDD").getOrCreate()

# Access the SparkContext
sc = spark.sparkContext
   
tsdata = [
    {'label': 1, 'time_series': [1.2, 2.4, 3.6, 4.8, 6.0]},
    {'label': 1, 'time_series': [1.0, 1.8, 2.6, 3.4, 4.2]},
    {'label': 1, 'time_series': [0.9, 1.8, 2.7, 3.6, 4.5]},
    {'label': 1, 'time_series': [1.5, 2.1, 2.7, 3.3, 3.9]},
    {'label': 1, 'time_series': [0.8, 1.7, 2.5, 3.2, 4.0]},
    {'label': 2, 'time_series': [2.1, 3.3, 4.5, 5.7, 6.9]},
    {'label': 2, 'time_series': [3.0, 3.8, 4.6, 5.4, 6.2]},
    {'label': 2, 'time_series': [3.3, 4.1, 4.9, 5.7, 6.5]},
    {'label': 3, 'time_series': [0.5, 1.5, 2.5, 3.5, 4.5]},
    {'label': 3, 'time_series': [2.0, 2.5, 3.0, 3.5, 4.0]},
    {'label': 4, 'time_series': [5.5, 6.6, 7.7, 8.8, 9.9]},
    {'label': 4, 'time_series': [6.1, 6.2, 6.3, 6.4, 6.5]},
    {'label': 1, 'time_series': [0.7, 1.3, 1.9, 2.5, 3.1]},
    {'label': 1, 'time_series': [1.1, 2.1, 3.1, 4.1, 5.1]},
    {'label': 1, 'time_series': [0.6, 1.2, 1.8, 2.4, 3.0]},
    {'label': 2, 'time_series': [2.4, 3.5, 4.6, 5.7, 6.8]},
    {'label': 2, 'time_series': [1.9, 2.8, 3.7, 4.6, 5.5]},
    {'label': 3, 'time_series': [1.0, 1.8, 2.6, 3.4, 4.2]},
    {'label': 4, 'time_series': [6.0, 7.0, 8.0, 9.0, 10.0]},
    {'label': 1, 'time_series': [1.3, 2.3, 3.3, 4.3, 5.3]}
]

df = spark.createDataFrame(tsdata)

global_model = GlobalModelManager()
global_model.train(df)