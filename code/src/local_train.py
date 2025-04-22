from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import random
from collections import Counter
from temp import DISTANCE_MEASURES_



# Node class for the Proximity Tree
class ProximityNode:
    def __init__(self, measure=None, exemplars=None, branches=None, label=None):
        self.measure = measure        # distance function
        self.exemplars = exemplars    # one exemplar per class
        self.branches = branches      # child nodes {class_label: ProximityNode}
        self.label = label            # class label if this is a leaf

# The Proximity Tree learner
class ProximityTree:
    def __init__(self, r=2):
        self.r = r  # number of candidate splits
        self.root = None

    def fit(self, X, y):
        data = list(zip(X, y))
        self.root = self._build_tree(data)

    def _build_tree(self, data):
        labels = [label for _, label in data]
        # Base case: pure node
        if len(set(labels)) == 1:
            return ProximityNode(label=labels[0])

        best_split = None
        best_impurity = -float('inf')

        # Try r candidate splits
        for _ in range(self.r):
            measure = random.choice(DISTANCE_MEASURES_)
            class_exemplars = self._select_random_exemplars(data)
            split = self._split_data(data, measure, class_exemplars)

            gini_gain = self._gini_gain(labels, split)
            if gini_gain > best_impurity:
                best_impurity = gini_gain
                best_split = (measure, class_exemplars, split)

        # Create child branches
        measure, exemplars, split_data = best_split
        branches = {}
        for exemplar, subset in split_data.items():
            if subset:  # If a branch has data
                branches[exemplar] = self._build_tree(subset)

        return ProximityNode(measure=measure, exemplars=exemplars, branches=branches)

    def _select_random_exemplars(self, data):
        class_dict = {}
        for ts, label in data:
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(ts)
        # Random exemplar for each class
        return {label: random.choice(ts_list) for label, ts_list in class_dict.items()}

    def _split_data(self, data, measure, exemplars):
        split = {label: [] for label in exemplars}
        for ts, label in data:
            # Assign to closest exemplar
            distances = {label_: measure(ts, ex) for label_, ex in exemplars.items()}
            closest_label = min(distances, key=distances.get)
            split[closest_label].append((ts, label))
        return split

    def _gini_gain(self, parent_labels, split):
        def gini(labels):
            count = Counter(labels)
            probs = [c / len(labels) for c in count.values()]
            return 1 - sum(p**2 for p in probs)

        parent_gini = gini(parent_labels)
        total = len(parent_labels)
        weighted_child_gini = sum(
            (len(branch) / total) * gini([label for _, label in branch])
            for branch in split.values() if branch
        )
        return parent_gini - weighted_child_gini

    def predict(self, ts):
        return self._predict_node(ts, self.root)

    def _predict_node(self, ts, node):
        if node.label is not None:
            return node.label

        # Find closest exemplar
        distances = {label: node.measure(ts, ex) for label, ex in node.exemplars.items()}
        closest_label = min(distances, key=distances.get)
        return self._predict_node(ts, node.branches[closest_label])
    
    

class ProximityForest:
    def __init__(self, n_trees=10, r=5):
        self.n_trees = n_trees
        self.r = r
        self.trees = [ProximityTree(r) for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, ts):
        predictions = [tree.predict(ts) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]
    
    


# Initialize Spark
spark = SparkSession.builder \
    .appName("ProximityForestDistributed") \
    .getOrCreate()

sc = spark.sparkContext

sc.addPyFile("code/src/temp.py")

# Load your data using Pandas, then parallelize
df = pd.read_csv('fulldataset_ECG5000.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Combine X and y for Spark
data = list(zip(X, y))
rdd = sc.parallelize(data, numSlices=4)  # Split into 4 partitions

# Split each partition into train/test locally
def split_partition(partition):
    partition = list(partition)
    random.shuffle(partition)
    split = int(len(partition) * 0.8)
    train = partition[:split]
    test = partition[split:]
    return [(train, test)]

split_rdd = rdd.mapPartitions(split_partition)

def train_tree(partition_data):
    train_data, _ = partition_data
    X_train, y_train = zip(*train_data)
    tree = ProximityTree(r=20)
    tree.fit(list(X_train), list(y_train))
    return [tree]

trained_trees = split_rdd.flatMap(train_tree).collect()


# Assemble into a forest at the driver
distributed_forest = ProximityForest(n_trees=len(trained_trees))
distributed_forest.trees = trained_trees

# Broadcast the forest back to all workers
broadcast_forest = sc.broadcast(distributed_forest)

# Predict on the test set in each partition
def predict_partition(partition_data):
    _, test_data = partition_data
    forest = broadcast_forest.value
    predictions = []
    for ts, true_label in test_data:
        pred = forest.predict(ts)
        predictions.append((true_label, pred))
    return predictions

results = split_rdd.flatMap(predict_partition).collect()

# Separate labels for evaluation
y_test, y_pred = zip(*results)

# Print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Stop Spark session
spark.stop()











'''


from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('fulldataset_ECG5000.csv', delimiter=',')

X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Lengths:\n  X_train:{len(X_train)}\n  y_train:{len(y_train)}\n  X_test:{len(X_test)}\n  y_test:{len(y_test)}')



# Initialize and train a Proximity Forest
forest = ProximityForest(n_trees=5, r=20)
forest.fit(X_train, y_train)

# Predict and print results
for i, ts in enumerate(X_test):
    predicted = forest.predict(ts)
    #print(f"Test Time Series {i+1}: Predicted Class = {predicted}")
    
    

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the test set
y_pred = [forest.predict(ts) for ts in X_test]

# Print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


'''