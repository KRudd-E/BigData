# evaluation.py
"""


Computes performance metrics for the model pipeline including:
- Classification accuracy
- Runtime per stage (data load, preprocessing, training, prediction)
- Scale-up efficiency metrics
"""
from pyspark.sql import functions as F 
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import  DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import logging
import numpy as np
from typing import Dict, Any
import collections
from typing import Dict, Any, Tuple
from aeon.classification.distance_based import  ProximityForest

# Define TreeNode namedtuple at the module level
TreeNode = collections.namedtuple(
    "TreeNode",
    "node_id parent_id split_on is_leaf prediction children".split()
)

class Evaluator:
    def __init__(self, decimal_precision: int = 6, track_memory: bool = False):
        self.metrics = {}
        self.start_times = {}
        self.precision = decimal_precision
        self.track_memory = track_memory
        self.class_labels: Dict[int, str] = {
            1: "Normal Beat (N)",
            2: "R-on-T PVC (r)",
            3: "Supraventricular Beat (S)",
            4: "PVC (V)",
            5: "Unclassifiable (Q)"
        }
        self.logger = logging.getLogger(__name__)

        # Configure logging only if no handlers are present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start_timer(self, stage_name: str):
        """Start timing for a pipeline stage"""
        self.start_times[stage_name] = time.time()
        if self.track_memory:
            self.record_memory_usage(f"start_{stage_name}")

    def record_time(self, stage_name: str, units: str = "seconds"):
        """Record duration with configurable units"""
        if stage_name not in self.start_times:
            raise ValueError(f"No timer started for: {stage_name}")

        duration = time.time() - self.start_times[stage_name]
        if units == "minutes":
            duration = duration / 60
        elif units == "hours":
            duration = duration / 3600

        self.metrics[f"{stage_name}_time"] = round(duration, 4)
        self.metrics[f"{stage_name}_time_units"] = units
        self.logger.info(f"{stage_name} took {duration:.2f}{units[0]}")

        if self.track_memory:
            self.record_memory_usage(f"end_{stage_name}")
        return duration

    def record_memory_usage(self, stage: str = None):
        """Record memory usage if running on Spark"""
        try:
            from pyspark import SparkContext
            # Attempt to get executor memory status if SparkContext is available
            sc = SparkContext.getOrCreate()
            if sc:
                 memory_usage = sc.getExecutorMemoryStatus()
                 key = "memory_usage" if not stage else f"memory_usage_{stage}"
                 self.metrics[key] = memory_usage
            else:
                 self.logger.warning("SparkContext not available to measure memory usage.")

        except Exception as e:
            # Catch any exception during memory measurement
            self.logger.warning(f"Could not measure memory usage: {e}")


    def calculate_classification_metrics(self, predictions_df: DataFrame) -> Dict:
        """Calculate multiple classification metrics"""
        metrics = {}
        # Ensure 'label' and 'prediction' columns exist before evaluating
        if "label" not in predictions_df.columns or "prediction" not in predictions_df.columns:
            self.logger.warning("Cannot calculate classification metrics: 'label' or 'prediction' column missing.")
            return metrics # Return empty metrics

        # Ensure label and prediction columns are numeric for the evaluator
        # Cast to DoubleType as MulticlassClassificationEvaluator expects numeric types
        predictions_df = predictions_df.withColumn("label", F.col("label").cast(DoubleType()))
        predictions_df = predictions_df.withColumn("prediction", F.col("prediction").cast(DoubleType()))


        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )

        # List of metrics to calculate
        metrics_to_calculate = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

        for metric in metrics_to_calculate:
            try:
                metrics[metric] = evaluator.evaluate(
                    predictions_df,
                    {evaluator.metricName: metric}
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate metric '{metric}': {e}")
                metrics[metric] = None # Indicate failure

        # Store calculated metrics, rounding to specified precision
        self.metrics.update({k: round(v, self.precision) if v is not None else None for k, v in metrics.items()})
        return metrics

    def calculate_model_complexity(self, model) -> None:
        """Calculate tree depth/size metrics for the provided model."""

        # Check if it's an AEON ProximityForest ensemble
        if isinstance(model, ProximityForest) and hasattr(model, "trees_"):
            self.logger.info("Calculating complexity for AEON ProximityForest ensemble.")
            depths, leaf_counts, split_counts = [], [], []

            for tree in model.trees_: # Iterate over individual trees in the ensemble
                try:
                    # AEON trees have a 'root' attribute
                    if hasattr(tree, "root"):
                        # Use existing AEON helper methods for traversal
                        depth = self._tree_depth(tree.root)
                        depths.append(depth)

                        leaves = self._count_leaves(tree.root)
                        leaf_counts.append(leaves)

                        splits = self._count_splits(tree.root)
                        split_counts.append(splits)

                        self.logger.debug(f"AEON Tree - Depth: {depth}, Leaves: {leaves}, Splits: {splits}")
                except Exception as e:
                    self.logger.warning(f"Error analyzing an AEON tree: {str(e)}")
                    continue # Continue to the next tree even if one fails

            # Store ensemble metrics (averages)
            self.metrics.update({
                "model_type": "local_ensemble", # Indicate model type
                "num_trees": len(model.trees_),
                "avg_depth": float(np.mean(depths)) if depths else 0.0,
                "avg_leaves": float(np.mean(leaf_counts)) if leaf_counts else 0.0,
                "avg_splits": float(np.mean(split_counts)) if split_counts else 0.0
            })

        # Check if it's a GlobalProxTree 
        # Assuming GlobalProxTree has a 'tree' attribute which is the dictionary
        # and the root node is at key 0.
        elif hasattr(model, 'tree') and isinstance(model.tree, dict) and 0 in model.tree:
             self.logger.info("Calculating complexity for Global Spark ProximityTree.")
             tree_dict = model.tree
             root_node_id = 0 # Assuming root is always 0

             try:
                 # Calculate complexity for the single global tree using new helper methods
                 depth = self._global_tree_depth(tree_dict, root_node_id)
                 leaves = self._global_count_leaves(tree_dict, root_node_id)
                 splits = self._global_count_splits(tree_dict, root_node_id)

                 # Store single tree metrics directly
                 self.metrics.update({
                     "model_type": "global_tree", # Indicate model type
                     "depth": depth,
                     "leaves": leaves,
                     "splits": splits,
                     "num_trees": 1 # It's a single tree
                 })
                 self.logger.debug(f"Global Tree - Depth: {depth}, Leaves: {leaves}, Splits: {splits}")

             except Exception as e:
                 self.logger.warning(f"Error analyzing GlobalProxTree structure: {str(e)}")
                 # Add placeholder metrics if analysis fails
                 self.metrics.update({
                     "model_type": "global_tree",
                     "depth": "Error",
                     "leaves": "Error",
                     "splits": "Error",
                     "num_trees": 1
                 })

        else:
            # If the model type is not recognized for complexity calculation
            self.logger.warning("Provided object is not a recognized model type for complexity metrics.")
            self.metrics.update({"model_type": "unrecognized", "complexity": "N/A"})


    # --- Helper methods for AEON ProximityTree traversal (existing) ---
    def _tree_depth(self, node) -> int:
        """Recursively calculate depth of an AEON ProximityTree node."""
        # Check if the node has children and if the children dictionary is not empty
        if not hasattr(node, 'children') or not node.children:
            return 1 # Base case: leaf node has depth 1
        # Recursive step: max depth of children + 1 for the current node
        return 1 + max((self._tree_depth(child)
                        for child in node.children.values()), default=0) # Use default=0 for empty children list

    def _count_leaves(self, node) -> int:
        """Count leaf nodes in an AEON ProximityTree."""
        # Check if the node has children and if the children dictionary is not empty
        if not hasattr(node, 'children') or not node.children:
            return 1 # Base case: this node is a leaf
        # Recursive step: sum the leaves in all child nodes
        return sum(self._count_leaves(child)
                   for child in node.children.values())

    def _count_splits(self, node) -> int:
        """Count split nodes (internal nodes) in an AEON ProximityTree."""
        # Check if the node has children and if the children dictionary is not empty
        if not hasattr(node, 'children') or not node.children:
            return 0 # Base case: this is a leaf, not a split node
        # Recursive step: this node is a split node (1) + sum of splits in children
        return 1 + sum(self._count_splits(child)
                       for child in node.children.values())

    # --- Helper methods for GlobalProxTree traversal  ---
    def _global_tree_depth(self, tree_dict: Dict[int, TreeNode], node_id: int) -> int:
        """Recursively calculate depth of a GlobalProxTree (from its dict representation)"""
        # Check if node_id exists in the dictionary and if it has children
        if node_id not in tree_dict or not tree_dict[node_id].children:
            return 1 # Base case: leaf node has depth 1
        # Recursively find the max depth of children + 1 for the current node
        return 1 + max((self._global_tree_depth(tree_dict, child_id)
                        for child_id in tree_dict[node_id].children.values()), default=0) # Use default=0 for empty children dict


    def _global_count_leaves(self, tree_dict: Dict[int, TreeNode], node_id: int) -> int:
        """Count leaf nodes in a GlobalProxTree (from its dict representation)"""
        # Check if node_id exists in the dictionary and if it has children
        if node_id not in tree_dict or not tree_dict[node_id].children:
            return 1 # Base case: this node is a leaf
        # Recursive step: sum the leaves in all child nodes
        return sum(self._global_count_leaves(tree_dict, child_id)
                   for child_id in tree_dict[node_id].children.values())


    def _global_count_splits(self, tree_dict: Dict[int, TreeNode], node_id: int) -> int:
        """Count split nodes (internal nodes) in a GlobalProxTree (from its dict representation)"""
        # Check if node_id exists in the dictionary and if it has children
        if node_id not in tree_dict or not tree_dict[node_id].children:
            return 0 # Base case: this is a leaf, not a split node
        # Recursive step: this node is a split node (1) + sum of splits in children
        return 1 + sum(self._global_count_splits(tree_dict, child_id)
                       for child_id in tree_dict[node_id].children.values())


    def generate_report(self, format: str = "dict", decimal_precision: int = 8) -> Any:
        """Generate report in multiple formats with configurable decimal precision

        Args:
            format: Output format - 'dict', 'json', or 'dataframe'
            decimal_precision: Number of decimal places to show for numeric values

        Returns:
            Report in requested format with high precision numeric values
        """
        # Helper function to format numeric values with precision
        def format_value(val, precision):
            if isinstance(val, (int, float)):
                return float(f"{val:.{precision}f}")
            return val

        report = {
            "performance": {
                "accuracy": format_value(self.metrics.get("accuracy", None), decimal_precision),
                "error_rate": format_value(1 - self.metrics.get("accuracy", 0), decimal_precision),
                "precision": format_value(self.metrics.get("weightedPrecision", None), decimal_precision),
                "recall": format_value(self.metrics.get("weightedRecall", None), decimal_precision),
                "f1_score": format_value(self.metrics.get("f1", None), decimal_precision)
            },
            "timing": {
                k: format_value(v, decimal_precision)
                for k, v in self.metrics.items()
                if "_time" in k and not k.endswith("_units")
            },
            "complexity": {}, # Initialize as empty dict

            "confusion_matrix": self.metrics.get("confusion_matrix", []),

            "class_wise": {
                 # Ensure class_wise metrics are formatted
                 k: {sub_k: format_value(sub_v, decimal_precision) for sub_k, sub_v in v.items()}
                 for k, v in self.metrics.get("class_wise", {}).items()
             },


            "_meta": {
                "decimal_precision": decimal_precision,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Populate complexity based on what was calculated in calculate_model_complexity
        model_type = self.metrics.get("model_type", "unknown")
        report["complexity"]["model_type"] = model_type

        if model_type == "local_ensemble":
            report["complexity"]["num_trees"] = int(self.metrics.get("num_trees", 0))
            report["complexity"]["avg_depth"] = format_value(self.metrics.get("avg_depth", 0), decimal_precision)
            report["complexity"]["avg_leaves"] = format_value(self.metrics.get("avg_leaves", 0), decimal_precision)
            report["complexity"]["avg_splits"] = format_value(self.metrics.get("avg_splits", 0), decimal_precision)
        elif model_type == "global_tree":
            report["complexity"]["num_trees"] = int(self.metrics.get("num_trees", 1)) # Should be 1
            report["complexity"]["depth"] = self.metrics.get("depth", "N/A") # Single depth
            report["complexity"]["leaves"] = self.metrics.get("leaves", "N/A") # Single leaves count
            report["complexity"]["splits"] = self.metrics.get("splits", "N/A") # Single splits count
        else:
            report["complexity"]["message"] = "Complexity metrics not applicable or calculated."


        # Handle JSON serialization
        if format == "json":
            import json
            # Use the default encoder, as format_value already handles floats
            return json.dumps(report, indent=2)

        # Handle DataFrame conversion
        elif format == "dataframe":
            import pandas as pd
            # Flatten the report dictionary for DataFrame conversion
            flattened_data = []
            for category, metrics_dict in report.items():
                if category in ["performance", "timing", "complexity"]:
                     if category == "complexity" and metrics_dict.get("model_type") == "local_ensemble":
                          # Flatten local ensemble metrics
                          for metric_name, value in metrics_dict.items():
                               if metric_name != "model_type":
                                    flattened_data.append({'category': category, 'metric': metric_name, 'value': value})
                     elif category == "complexity" and metrics_dict.get("model_type") == "global_tree":
                          # Flatten global tree metrics
                          for metric_name, value in metrics_dict.items():
                               if metric_name != "model_type":
                                    flattened_data.append({'category': category, 'metric': metric_name, 'value': value})
                     elif category in ["performance", "timing"]:
                          for metric_name, value in metrics_dict.items():
                                flattened_data.append({'category': category, 'metric': metric_name, 'value': value})
                elif category == "class_wise":
                     for class_key, metrics_dict in metrics_dict.items():
                          for metric_name, value in metrics_dict.items():
                                flattened_data.append({'category': f"class_wise_{class_key}", 'metric': metric_name, 'value': value})
                # Skip _meta and confusion_matrix for this simple flattening


            df = pd.DataFrame(flattened_data)
            return df

        return report


    def log_metrics(self, predictions_df: DataFrame, model=None) -> Tuple[Dict, Dict]:
        """
        Run complete evaluation pipeline, calculate metrics, and log them.

        Args:
            predictions_df: Spark DataFrame with 'label' and 'prediction' columns.
            model: The trained model object (AEON ensemble or GlobalProxTree),
                   used for complexity metrics.

        Returns:
            Tuple[Dict, Dict]: The full report dictionary and the class labels dictionary.
        """
        self.logger.info("\n=== Calculating Evaluation Metrics ===")

        # Calculate classification metrics (works for both model types)
        self.calculate_classification_metrics(predictions_df)

        # Calculate confusion matrix (uncomment if needed)
        # self._log_confusion_matrix(predictions_df)

        # Calculate model complexity (handles both model types)
        if model:
            self.calculate_model_complexity(model)
        else:
             self.logger.warning("No model provided for complexity metrics.")

        # Calculate class-wise metrics (uncomment if needed)
        # self._log_class_wise_metrics(predictions_df)


        # Generate the final report dictionary
        report = self.generate_report()

        # Log performance metrics
        self.logger.info("\n=== Performance Metrics ===")
        performance_metrics = report.get("performance", {})
        if performance_metrics:
            for metric, value in performance_metrics.items():
                if value is not None:
                    # Use the precision from the report metadata for logging
                    log_precision = report["_meta"].get("decimal_precision", self.precision)
                    self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.{log_precision}f}")
                else:
                     self.logger.info(f"{metric.replace('_', ' ').title()}: N/A")
        else:
            self.logger.info("No performance metrics calculated.")


        # Log timing metrics
        self.logger.info("\n=== Timing Metrics ===")
        timing_metrics = report.get("timing", {})
        if timing_metrics:
            for stage, t in timing_metrics.items():
                try:
                    # Ensure t is treated as a float for logging format
                    t_float = float(t)
                    units = self.metrics.get(f"{stage}_units", "s") # Get units from self.metrics
                    self.logger.info(f"{stage.replace('_time', '').replace('_', ' ').title()}: {t_float:.2f}{units}")
                except (ValueError, TypeError):
                    # Fallback if conversion fails
                    self.logger.info(f"{stage.replace('_time', '').replace('_', ' ').title()}: {t} (no units available)")
        else:
            self.logger.info("No timing metrics recorded.")


        # Log complexity metrics
        complexity_metrics = report.get("complexity", {})
        if complexity_metrics:
             self.logger.info("\n=== Model Complexity ===")
             model_type = complexity_metrics.get("model_type", "unknown")
             self.logger.info(f"Model Type: {model_type}")

             if model_type == "local_ensemble":
                 self.logger.info(f"Number of Trees: {complexity_metrics.get('num_trees', 'N/A')}")
                 self.logger.info(f"Average Depth: {complexity_metrics.get('avg_depth', 'N/A')}")
                 self.logger.info(f"Average Leaves: {complexity_metrics.get('avg_leaves', 'N/A')}")
                 self.logger.info(f"Average Splits: {complexity_metrics.get('avg_splits', 'N/A')}")
             elif model_type == "global_tree":
                  self.logger.info(f"Number of Trees: {complexity_metrics.get('num_trees', 'N/A')}")
                  self.logger.info(f"Depth: {complexity_metrics.get('depth', 'N/A')}")
                  self.logger.info(f"Leaves: {complexity_metrics.get('leaves', 'N/A')}")
                  self.logger.info(f"Splits: {complexity_metrics.get('splits', 'N/A')}")
             else:
                  self.logger.info(complexity_metrics.get("message", "Complexity metrics not available."))
        else:
            self.logger.info("No complexity metrics calculated.")


        # Log class-wise metrics (uncomment if needed)
        # class_wise_metrics = report.get("class_wise", {})
        # if class_wise_metrics:
        #     self.logger.info("\n=== Class-wise Metrics ===")
        #     for class_key, metrics_dict in class_wise_metrics.items():
        #          class_label = class_key.replace("class_", "") # Extract label from key
        #          self.logger.info(f"--- Class {class_label} ---")
        #          for metric_name, value in metrics_dict.items():
        #               if value is not None:
        #                    log_precision = report["_meta"].get("decimal_precision", self.precision)
        #                    self.logger.info(f"  {metric_name.title()}: {value:.{log_precision}f}")
        #               else:
        #                    self.logger.info(f"  {metric_name.title()}: N/A")


        return report, self.class_labels


    def _log_confusion_matrix(self, predictions_df):
        """
        Log the confusion matrix by converting the predictions DataFrame
        into an RDD of (prediction, label) tuples and calculating metrics.
        """
        self.logger.info("\n=== Confusion Matrix ===")
        # Ensure 'label' and 'prediction' columns exist and are numeric
        if "label" not in predictions_df.columns or "prediction" not in predictions_df.columns:
            self.logger.warning("Cannot compute confusion matrix: 'label' or 'prediction' column missing.")
            return
        # Cast to integer for MulticlassMetrics
        prediction_and_labels_rdd = predictions_df.select(
            F.col("prediction").cast("integer"),
            F.col("label").cast("integer")
        ).rdd.map(tuple)

        # Debug: Check how many tuples you have and print a few
        count = prediction_and_labels_rdd.count()
        self.logger.debug(f"Count of (prediction, label) tuples for confusion matrix: {count}")

        # If the RDD is empty, log a warning and exit the function.
        if count == 0:
            self.logger.warning("No valid data for confusion matrix")
            return

        # Calculate confusion matrix using MLlib's MulticlassMetrics.
        try:
            metrics = MulticlassMetrics(prediction_and_labels_rdd)
            # Get the confusion matrix as a local NumPy array and convert to list of lists
            cm = metrics.confusionMatrix().toArray().tolist()
            self.metrics["confusion_matrix"] = cm
            # Log the confusion matrix (can be verbose for large matrices)
            # self.logger.info("Confusion Matrix:")
            # for row in cm:
            #     self.logger.info(row)
            self.logger.debug(f"Confusion matrix computed: {cm}")

        except Exception as e:
            self.logger.error(f"Failed to compute confusion matrix: {str(e)}")
            # Store error message in metrics if computation fails
            self.metrics["confusion_matrix"] = f"Error: {str(e)}"


    def _log_class_wise_metrics(self, predictions_df: DataFrame):
        """
        Log per-class metrics (precision, recall, f1) for each class using MLlib.
        """
        self.logger.info("\n=== Class-wise Metrics ===")
        # Convert predictions and labels to integer type and form an RDD of tuples
        # Ensure 'label' and 'prediction' columns exist and are numeric before casting
        if "label" not in predictions_df.columns or "prediction" not in predictions_df.columns:
            self.logger.warning("Cannot compute class-wise metrics: 'label' or 'prediction' column missing.")
            return
        prediction_and_labels_rdd = predictions_df.select(
            F.col("prediction").cast("integer"),
            F.col("label").cast("integer")
        ).rdd.map(tuple)

        # Debug: Count the number of (prediction, label) tuples
        count = prediction_and_labels_rdd.count()
        self.logger.debug(f"Total (prediction, label) pairs for class metrics: {count}")
        if count == 0:
            self.logger.warning("No valid data for class wise metrics")
            return

        # Compute metrics using MLlib
        try:
            metrics = MulticlassMetrics(prediction_and_labels_rdd)
        except Exception as e:
            self.logger.error(f"Error computing MLlib metrics for class-wise evaluation: {str(e)}")
            return # Exit if MLlib metrics computation fails

        # Get the unique classes from the DataFrame
        try:
            # Collect distinct labels from the original DataFrame to get class names
            # Use the original 'label' column before casting to integer if necessary
            distinct_labels = [row["label"] for row in predictions_df.select("label").distinct().collect()]
            # Ensure labels are integers and sort them
            classes = sorted([int(l) for l in distinct_labels if l is not None])
            self.logger.debug(f"Unique classes found for class-wise metrics: {classes}")
        except Exception as e:
            self.logger.error(f"Error fetching unique classes for class-wise metrics: {str(e)}")
            classes = [] # Set classes to empty list if fetching fails

        class_metrics = {}
        # For each class, calculate precision, recall, and f1 score
        for cls in classes:
            try:
                # MLlib metrics methods expect float labels
                class_float = float(cls)
                precision = metrics.precision(class_float)
                recall = metrics.recall(class_float)
                f1 = metrics.fMeasure(class_float, beta=1.0) # beta=1.0 for F1 score

                class_key = f"class_{cls}"
                class_metrics[class_key] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
                self.logger.debug(f"Metrics for class {cls}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
            except Exception as e:
                self.logger.warning(f"Class {cls} metrics error: {str(e)}")
                # Store error/None for this class if calculation fails
                class_metrics[f"class_{cls}"] = {"precision": None, "recall": None, "f1": None, "error": str(e)}

        # Save the computed metrics in our metrics dictionary
        self.metrics["class_wise"] = class_metrics

