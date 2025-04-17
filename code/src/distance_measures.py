'''file containing variety of distance based decision making methods to import into 
for now is all just as functions but could theoretically be turned into a class'''

import pickle
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from aeon.classification.distance_based import ProximityTree, ProximityForest
import logging
from random import sample
from dtaidistance import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def calc_dtw_distance(iterator):
        partition_data = list(iterator)
        updated_rows = []
        
        for row in partition_data:
            time_series = row['time_series']
            exemplars = row['exemplars']
            
            dtw_distances = [dtw.distance(time_series, exemplar) for exemplar in exemplars]
            
            updated_row = {**row}

            for i, dtw_distance in enumerate(dtw_distances):
                updated_row[f"exemplar_{i+1}"] = dtw_distance
            
            updated_rows.append(updated_row)
        
        return iter(updated_rows)
    
def calc_euclid_distance(iterator):
        partition_data = list(iterator)
        updated_rows = []
        
        for row in partition_data:
            time_series = row['time_series']
            exemplars = row['exemplars']
            
            dtw_distances = [dtw.distance(time_series, exemplar,only_ub=True) for exemplar in exemplars]
            
            updated_row = {**row}

            for i, dtw_distance in enumerate(dtw_distances):
                updated_row[f"exemplar_{i+1}"] = dtw_distance
            
            updated_rows.append(updated_row)
        
        return iter(updated_rows)
    
  
dummy_partition = iter([
    {
        'id': 1,
        'label': 'A',
        'time_series': [1, 2, 3, 4],
        'exemplars': [
            [1, 2, 2, 3],
            [2, 3, 4, 5]
        ]
    },
    {
        'id': 2,
        'label': 'B',
        'time_series': [5, 4, 3, 2],
        'exemplars': [
            [5, 5, 4, 3],
            [3, 3, 2, 1]
        ]
    }
])
#print('dtw_dist:')
#print(list(calc_dtw_distance(dummy_partition)))
print('euclid_dist:')
print(list(calc_euclid_distance(dummy_partition))) #for some reason wont calculate this if has already calculated dtw dist