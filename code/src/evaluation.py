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
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import logging
import numpy as np
from typing import Dict, Any


class Evaluator:
    def __init__(self, decimal_precision: int = 6, track_memory: bool = False):
        self.metrics = {}
        self.start_times = {}
        self.precision = decimal_precision
        self.track_memory = track_memory
        class_labels: Dict[int, str] = None  
        self.class_labels = class_labels or {
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
            memory_usage = SparkContext.getOrCreate().getExecutorMemoryStatus()
            key = "memory_usage" if not stage else f"memory_usage_{stage}"
            self.metrics[key] = memory_usage
        except:
            self.logger.warning("Could not measure memory usage")

    def calculate_classification_metrics(self, predictions_df: DataFrame) -> Dict:
        """Calculate multiple classification metrics"""
        metrics = {}
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction"
        )
        
        for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
            metrics[metric] = evaluator.evaluate(
                predictions_df, 
                {evaluator.metricName: metric}
            )
        
        self.metrics.update({k: round(v, self.precision) for k, v in metrics.items()})
        return metrics

    def calculate_model_complexity(self, ensemble) -> None:
        """Calculate tree depth/size metrics for Proximity Forest"""
        if not ensemble or not hasattr(ensemble, "trees_"):
            self.logger.warning("No valid ensemble for complexity metrics")
            return
            
        depths, leaf_counts, split_counts = [], [], []
        
        for tree in ensemble.trees_:
            try:
                if hasattr(tree, "root"):
                    # Calculate depth
                    depth = self._tree_depth(tree.root)
                    depths.append(depth)
                    
                    # Calculate leaves
                    leaves = self._count_leaves(tree.root)
                    leaf_counts.append(leaves)
                    
                    # Calculate splits
                    splits = self._count_splits(tree.root)
                    split_counts.append(splits)
                    
                    self.logger.debug(f"Tree - Depth: {depth}, Leaves: {leaves}, Splits: {splits}")
            except Exception as e:
                self.logger.warning(f"Error analyzing tree: {str(e)}")
                continue
                    
        self.metrics.update({
            "num_trees": len(ensemble.trees_),
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "avg_leaves": float(np.mean(leaf_counts)) if leaf_counts else 0.0,
            "avg_splits": float(np.mean(split_counts)) if split_counts else 0.0
        })
    
    def _tree_depth(self, node) -> int:
        """Recursively calculate depth of a ProximityTree"""
        if not hasattr(node, 'children') or not node.children:
            return 1
        return 1 + max((self._tree_depth(child) 
                    for child in node.children.values()), default=0)

    def _count_leaves(self, node) -> int:
        """Count leaf nodes in a ProximityTree"""
        if not hasattr(node, 'children') or not node.children:
            return 1
        return sum(self._count_leaves(child) 
                for child in node.children.values())

    def _count_splits(self, node) -> int:
        """Count split nodes in a ProximityTree"""
        if not hasattr(node, 'children') or not node.children:
            return 0
        return 1 + sum(self._count_splits(child)
                for child in node.children.values())
        
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
                "accuracy": format_value(self.metrics.get("accuracy", 0), decimal_precision),
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
            "complexity": {
                "num_trees": int(self.metrics.get("num_trees", 0)),
                "avg_depth": format_value(self.metrics.get("avg_depth", 0), decimal_precision),
                "avg_leaves": format_value(self.metrics.get("avg_leaves", 0), decimal_precision),
                "avg_splits": format_value(self.metrics.get("avg_splits", 0), decimal_precision)
            },
            
            "confusion_matrix": self.metrics.get("confusion_matrix", []),
            
            "class_wise": { 
                k: format_value(v, decimal_precision)
                for k, v in self.metrics.get("class_wise", {}).items()
            },
            
            "_meta": {
                "decimal_precision": decimal_precision,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Handle JSON serialization
        if format == "json":
            import json
            class DecimalEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, float):
                        return round(o, decimal_precision)
                    return super().default(o)
            return json.dumps(report, indent=2, cls=DecimalEncoder)
        
        # Handle DataFrame conversion
        elif format == "dataframe":
            import pandas as pd
            df = pd.DataFrame.from_dict({
                'metric': [k for group in report.values() for k in group.keys()],
                'value': [v for group in report.values() for v in group.values()],
                'category': [cat for cat in report.keys() 
                            for _ in range(len(report[cat]))]
            })
            return df
        
        return report
    
    def log_metrics(self, predictions_df: DataFrame, ensemble=None) -> Dict:
        """Run complete evaluation pipeline"""
        self.calculate_classification_metrics(predictions_df)
        #self._log_confusion_matrix(predictions_df)
        
        if ensemble:
            self.calculate_model_complexity(ensemble)
           #self._log_class_wise_metrics(predictions_df)
                
        report = self.generate_report()
        
        # Log performance metrics
        self.logger.info("\n=== Performance Metrics ===")
        for metric, value in report["performance"].items():
            if value is not None:
                self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.{self.precision}f}")
        
        # Log timing metrics
        self.logger.info("\n=== Timing Metrics ===")
        for stage, t in report["timing"].items():
            try:
                # Ensure t is treated as a float
                t_float = float(t)
                units = self.metrics.get(f"{stage}_units", "s")
                self.logger.info(f"{stage.replace('_time', '')}: {t_float:.2f}{units}")
            except (ValueError, TypeError):
                # Fallback if conversion fails
                self.logger.info(f"{stage.replace('_time', '')}: {t} (no units available)")
        
        # Log complexity metrics
        if ensemble:
            self.logger.info("\n=== Model Complexity ===")
            for metric, value in report["complexity"].items():
                self.logger.info(f"{metric.replace('_', ' ').title()}: {value}")
        
        return report, self.class_labels
    
    def _log_confusion_matrix(self, predictions_df):
        
        #Log confusion matrix
        prediction_and_labels = predictions_df.select(
            F.col("prediction").cast("integer"), 
            F.col("label").cast("integer")
        ).rdd.map(tuple)
        
        if prediction_and_labels.isEmpty():
            self.logger.warning("No valid data for confusion matrix")
            return
        metrics = MulticlassMetrics(prediction_and_labels)
        self.metrics["confusion_matrix"] = metrics.confusionMatrix().toArray().tolist()

    def _log_class_wise_metrics(self, predictions_df: DataFrame):

        #Log per-class metrics        
        prediction_and_labels = predictions_df.select(
            F.col("prediction").cast("integer"), 
            F.col("label").cast("integer")
        ).rdd.map(tuple)
        
        if prediction_and_labels.isEmpty():
            self.logger.warning("No valid data for class wise metrics plot")
            return
        
        metrics = MulticlassMetrics(prediction_and_labels)
        
        # Get unique classes from the DataFrame instead of the matrix
        classes = [float(row["label"]) for row in predictions_df.select("label").distinct().collect()]
        classes = sorted([int(c) for c in classes])  # Convert to sorted integers
        
    
        class_metrics = {}
        for cls in classes:
            try:
                
                class_key = f"class_{cls}"
                class_metrics[class_key] = {
                    "precision": metrics.precision(float(cls)),
                    "recall": metrics.recall(float(cls)),
                    "f1": metrics.fMeasure(float(cls), beta=1.0)
                }
            except Exception as e:
                self.logger.warning(f"Class {cls} metrics error: {str(e)}")
        
        self.metrics["class_wise"] = class_metrics