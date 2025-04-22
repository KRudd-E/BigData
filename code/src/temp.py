import numpy as np
# Simple distance functions
def euclidean(ts1, ts2):
    return np.linalg.norm(np.array(ts1) - np.array(ts2))

def manhattan(ts1, ts2):
    return np.sum(np.abs(np.array(ts1) - np.array(ts2)))

# Pool of distance measures
DISTANCE_MEASURES_ = [euclidean, manhattan]
