# global_model_manager.py (Optimized Version - Gini Fix)
"""
Implements the GlobalModelManager class for training a distribution-friendly 
proximity tree using Spark DataFrames. Includes optimizations for UDFs, 
exemplar sampling, count reduction, and reproducibility.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import pickle
import random # Import random for seeding
import sys
import time
from typing import Any, Dict, List, Tuple # Adjusted typing imports

import numpy as np
import pandas as pd
from pyspark import StorageLevel # Import StorageLevel
from pyspark.sql import DataFrame, Row, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf # Import pandas_udf
from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType,
                               StructField, StructType)

# Configure logging for this module
# Using a specific logger name is good practice
logger_gmm = logging.getLogger("GlobalModelManager") 
# Ensure handler is added only once
if not logger_gmm.handlers:
     handler_gmm = logging.StreamHandler(sys.stdout) # Log to stdout
     formatter_gmm = logging.Formatter('%(asctime)s - GMM - %(levelname)s - %(message)s')
     handler_gmm.setFormatter(formatter_gmm)
     logger_gmm.addHandler(handler_gmm)
     if logger_gmm.level == logging.NOTSET:
          logger_gmm.setLevel(logging.INFO) # Set desired level
     logger_gmm.propagate = False # Prevent duplicate logs

# Suppress excessive logging from py4j and pyspark itself if needed
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)


try:
    import numpy as np
    _NP_GMM = True # Use a distinct name
except ImportError: 
    _NP_GMM = False

# Define TreeNode namedtuple (should be defined once globally or imported)
TreeNode = collections.namedtuple(
    "TreeNode", "node_id parent_id split_on is_leaf prediction children".split()
)

# Keep the original efficient euclidean distance function (renamed)
def _euclid_gmm(a, b): 
    """Fast Euclidean distance for python *or* NumPy inputs."""
    # Add basic type/length checks for robustness within UDFs
    if a is None or b is None: return float("inf")
    # Check if inputs are list-like and have length attribute
    len_a = len(a) if hasattr(a, '__len__') else -1
    len_b = len(b) if hasattr(b, '__len__') else -1
    if len_a != len_b or len_a == -1: return float("inf")
    
    if _NP_GMM:
        try:
            # Ensure inputs are numpy arrays for subtraction
            a_np = np.asarray(a, dtype=float); b_np = np.asarray(b, dtype=float)
            diff = a_np - b_np; dist = float(np.sqrt(np.dot(diff, diff)))
            return dist
        except Exception as e: 
            # Avoid logging excessively inside UDF, maybe log sample errors if needed
            # logger_gmm.error(f"Error in _euclid_gmm (NumPy): {e}") 
            return float("inf") 
    else: # Pure Python path
        try:
            dist = float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))) # Add float conversion
            return dist
        except Exception as e: 
            # logger_gmm.error(f"Error in _euclid_gmm (Python): {e}")
            return float("inf") 

# =============================================================================
# GlobalModelManager class (Optimised + Enhanced Prediction)
# =============================================================================

class GlobalModelManager:
    """
    Distribution-friendly proximity-tree learner using Spark DataFrames.
    
    Optimizations Included:
    P1: Pandas UDFs for routing and prediction.
    P1: Reduced redundant .count() actions in fit loop.
    P2: Distributed exemplar sampling using Window functions.
    P3: Seeded RNG for reproducibility.
    P2: Uses MEMORY_AND_DISK caching for intermediate DataFrames.
    """
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.logger = logging.getLogger("GlobalModelManager") 
        self.logger.info("Initializing GlobalModelManager.")
        if "tree_params" not in config: raise ValueError("Config must contain 'tree_params'.")
        
        p = config["tree_params"]
        self.spark = spark
        self.max_depth: int | None = p.get("max_depth") 
        self.min_samples: int = p.get("min_samples_split", 2) 
        self.k: int = p.get("n_splitters", 5) 
        # P3: Add random_state for reproducibility
        self.random_state: int | None = p.get("random_state") 
        #self._rng = random.Random(self.random_state) # Initialize RNG instance with seed
        
        self.tree: Dict[int, TreeNode] = {0: TreeNode(0, None, None, False, None, {})}
        self._next_id: int = 1
        self._maj: int = 1 # Default majority class
        self.logger.info(f"Initialized with max_depth={self.max_depth}, min_samples={self.min_samples}, k={self.k}, seed={self.random_state}")

    def _to_ts_df(self, df):
        """ Ensure DataFrame has (row_id, time_series[, true_label]). """
        self.logger.debug("Converting DataFrame to time series format.")
        lbl = None
        if "label" in df.columns: lbl = "label"
        elif "true_label" in df.columns: lbl = "true_label"
        else: self.logger.warning("No 'label' or 'true_label' column found.")
        
        # Add row_id if it doesn't exist
        if "row_id" not in df.columns:
            self.logger.debug("Adding row_id column.")
            df = df.withColumn("row_id", F.monotonically_increasing_id())
        else: # Ensure existing row_id is LongType
            if df.schema["row_id"].dataType != LongType():
                 self.logger.debug("Casting existing row_id to LongType.")
                 df = df.withColumn("row_id", F.col("row_id").cast(LongType()))

        # Check if time_series column already exists
        if "time_series" in df.columns:
            self.logger.debug("'time_series' column already exists.")
            # Ensure label column is named 'true_label' if it exists
            if lbl == "label": df = df.withColumnRenamed("label", "true_label")
            # Select necessary columns
            select_cols = ["row_id", "time_series"]
            if "true_label" in df.columns: select_cols.append("true_label")
            return df.select(*select_cols)

        # If time_series doesn't exist, create it
        cols_to_exclude = {"row_id"}
        if lbl: cols_to_exclude.add(lbl)
        
        feat_cols = [c for c in df.columns if c not in cols_to_exclude]
        if not feat_cols: raise ValueError("No feature columns found to create 'time_series'.")
        self.logger.debug(f"Creating 'time_series' from feature columns: {feat_cols}")
        
        select_exprs = [ F.col("row_id"), F.array(*[F.col(c) for c in feat_cols]).alias("time_series") ]
        if lbl: select_exprs.append(F.col(lbl).cast(IntegerType()).alias("true_label"))
        
        return df.select(*select_exprs)

    @staticmethod
    def _gini(counts: Dict[int, int]) -> float:
        """ Calculates the Gini impurity for a dictionary of class counts. """
        tot = sum(counts.values())
        if tot == 0: return 0.0
        return 1.0 - sum((c / tot) ** 2 for c in counts.values())

    def fit(self, df): 
        """Train the proximity tree."""
        self.logger.info("Starting GlobalProxTree fitting process.")
        # P2: Use MEMORY_AND_DISK caching
        ts_df = self._to_ts_df(df).persist(StorageLevel.MEMORY_AND_DISK) 
        initial_row_count = ts_df.count() # P1: Necessary count here
        self.logger.info(f"Input data converted and cached. Row count: {initial_row_count}")

        if initial_row_count == 0: 
             self.logger.warning("Input DataFrame is empty. Cannot train tree.")
             ts_df.unpersist(); return self

        # Determine majority class
        try: 
            maj_row = ts_df.groupBy("true_label").count().orderBy(F.desc("count")).first()
            if maj_row: self._maj = maj_row["true_label"] 
            self.logger.info(f"Overall majority class: {self._maj}")
        except Exception as e: self.logger.error(f"Error calculating majority class: {e}. Using default: {self._maj}")

        # Initialize assignment DataFrame
        assign = ts_df.select("row_id", "time_series", "true_label") \
                      .withColumn("node_id", F.lit(0)) \
                      .persist(StorageLevel.MEMORY_AND_DISK) # P2: Use MEMORY_AND_DISK
        assign_count = assign.count(); self.logger.info(f"Initial assignment created. Rows: {assign_count}") # P1: Necessary count
        ts_df.unpersist() 

        # --- Tree Building Loop ---
        open_nodes = {0} 
        depth = 0
        while open_nodes and (self.max_depth is None or depth < self.max_depth):
            self.logger.info(f"--- Starting Tree Level {depth} ---")
            self.logger.debug(f"Open nodes: {open_nodes}")

            # Filter data for current level nodes
            cur = assign.filter(F.col("node_id").isin(list(open_nodes))) \
                        .persist(StorageLevel.MEMORY_AND_DISK) # P2: Use MEMORY_AND_DISK
            
            # --- P1: Calculate node stats ONCE per level ---
            self.logger.debug("Calculating statistics for current level nodes...")
            node_stats_df = cur.groupBy("node_id", "true_label").count()
            node_stats_rows = node_stats_df.collect() # Collect stats (expect relatively small)
            stats_per_node = collections.defaultdict(dict)
            totals_per_node = collections.defaultdict(int)
            for r in node_stats_rows:
                 stats_per_node[r.node_id][r.true_label] = r["count"]
                 totals_per_node[r.node_id] += r["count"]
            self.logger.debug(f"Calculated stats for {len(totals_per_node)} nodes.")
            # --- End P1 ---

            # Check if any data remains for open nodes (using precalculated totals)
            if not any(totals_per_node.get(nid, 0) > 0 for nid in open_nodes):
                 self.logger.info(f"No data for open nodes at depth {depth}. Stopping tree growth.")
                 cur.unpersist(); break

            # --- P2: Distributed Exemplar Sampling ---
            self.logger.debug("Starting distributed exemplar sampling...")
            # P3: Seed rand with instance RNG state (requires converting int to seed)
            window_spec = Window.partitionBy("node_id", "true_label").orderBy(F.rand()) # REMOVED SEED 
#            window_spec = Window.partitionBy("node_id", "true_label").orderBy(F.rand(self._rng.randint(0, 1000000))) 
            sampled_exemplars_df = cur.withColumn("rank", F.row_number().over(window_spec)) \
                                      .filter(F.col("rank") <= self.k) \
                                      .select("node_id", "true_label", "time_series") 
            collected_exemplars = sampled_exemplars_df.collect() # Collect ONLY the k*nodes*labels samples
            pool: Dict[int, Dict[int, list]] = collections.defaultdict(dict)
            for row in collected_exemplars: pool[row.node_id].setdefault(row.true_label, []).append(row.time_series) 
            self.logger.debug(f"Finished exemplar sampling. Nodes with pools: {list(pool.keys())}")
            # --- End P2 ---

            best_splits: Dict[int, Tuple[str, Dict[int, list]]] = {} 
            nodes_to_make_leaf: set[int] = set()

            # --- Evaluate splits (Driver-side logic using precalculated stats) ---
            self.logger.debug("Evaluating splits...")
            for nid in list(open_nodes): 
                self.logger.debug(f"Evaluating node {nid}...")
                stats = stats_per_node.get(nid, {})
                tot_samples_in_node = totals_per_node.get(nid, 0) # P1: Reuse count
                self.logger.debug(f"Node {nid} stats: {stats}, total samples: {tot_samples_in_node}")

                # Leaf conditions (using reused count)
                is_leaf = False
                if tot_samples_in_node < self.min_samples: is_leaf = True; reason="min_samples"
                elif len(stats) <= 1: is_leaf = True; reason="pure"
                elif nid not in pool or not pool[nid] or len(pool[nid]) < 2: is_leaf = True; reason="exemplars"
                
                if is_leaf: self.logger.info(f"Node {nid} becoming leaf: {reason}."); nodes_to_make_leaf.add(nid); continue

                # Find best split for non-leaf node
                parent_gini = self._gini(stats)
                best_gain = -1.0; best_exemplars_for_split = None
                node_pool = pool[nid]; available_labels = list(node_pool.keys())

                self.logger.debug(f"Node {nid}: Evaluating {self.k} candidates. Parent Gini: {parent_gini:.4f}")
                for k_idx in range(self.k):
                    candidate_ex = {}
                    possible = True
                    for lbl in available_labels:
                        if node_pool[lbl]: 
                            #candidate_ex[lbl] = self._rng.choice(node_pool[lbl]) # P3: Use seeded RNG
                            candidate_ex[lbl] = random.choice(node_pool[lbl]) 
                        else: possible = False; break 
                    if not possible or len(candidate_ex) < 2: continue 

                    bc_ex = self.spark.sparkContext.broadcast(candidate_ex)
                    
                    # Standard UDF for Gini calculation step (Pandas UDF less obvious benefit here)
                    @F.udf(IntegerType())
                    def nearest_lbl_udf_local(ts):
                        ex_val = bc_ex.value; best_d, best_l = float("inf"), None
                        for l, ex_ts in ex_val.items():
                            d = _euclid_gmm(ts, ex_ts); 
                            if d < best_d: best_d, best_l = d, l
                        return best_l

                    # Filter data for the current node *before* applying UDF
                    node_data_df = cur.filter(F.col("node_id") == nid) 
                    
                    # Calculate weighted Gini impurity (DataFrame based)
                    split_impurity_df = node_data_df.withColumn("branch", nearest_lbl_udf_local("time_series")) \
                                               .groupBy("branch", "true_label").count()
                    branch_totals = split_impurity_df.groupBy("branch").agg(F.sum("count").alias("branch_total"))
                    gini_per_branch = split_impurity_df.join(branch_totals, "branch") \
                                             .withColumn("prob_sq", (F.col("count") / F.col("branch_total")) ** 2) \
                                             .groupBy("branch", "branch_total").agg(F.sum("prob_sq").alias("s")) \
                                             .withColumn("branch_gini", 1.0 - F.col("s")) # <-- Corrected: sum("prob_sq")
                    weighted_gini_row = gini_per_branch.withColumn("weighted_gini", (F.col("branch_total") / tot_samples_in_node) * F.col("branch_gini")) \
                                                  .agg(F.sum("weighted_gini").alias("total_weighted_gini")) \
                                                  .first()
                    bc_ex.unpersist(False) 

                    if weighted_gini_row and weighted_gini_row["total_weighted_gini"] is not None:
                        current_impurity = weighted_gini_row["total_weighted_gini"]
                        current_gain = parent_gini - current_impurity
                        self.logger.debug(f"Node {nid}, Candidate {k_idx+1}: Impurity={current_impurity:.4f}, Gain={current_gain:.4f}")
                        if current_gain > best_gain:
                            best_gain = current_gain; best_exemplars_for_split = candidate_ex
                            self.logger.debug(f"Node {nid}: New best split found (Gain: {best_gain:.4f})")
                    else: self.logger.warning(f"Node {nid}, Candidate {k_idx+1}: Could not calculate impurity.")

                # Decide if node becomes leaf
                if best_gain <= 1e-9: 
                    self.logger.info(f"Node {nid} becoming leaf: best gain ({best_gain:.4f}) too low.")
                    nodes_to_make_leaf.add(nid)
                else:
                    self.logger.info(f"Node {nid}: Selected best split with gain {best_gain:.4f}.")
                    best_splits[nid] = ("euclidean", best_exemplars_for_split)

            # --- Finalize leaves ---
            self.logger.debug(f"Nodes to finalize as leaves: {nodes_to_make_leaf}")
            for nid in list(nodes_to_make_leaf): 
                if nid in open_nodes: 
                    stats = stats_per_node.get(nid, {}) # P1: Reuse stats
                    maj_lbl = self._maj 
                    if stats: maj_lbl = max(stats.items(), key=lambda kv: (kv[1], -kv[0]))[0] 
                    self.tree[nid] = self.tree[nid]._replace(is_leaf=True, prediction=maj_lbl, children={}, split_on=None)
                    self.logger.info(f"Node {nid} finalized as leaf. Prediction: {maj_lbl}.")
                    open_nodes.remove(nid) 

            # --- Create children and update assignments ---
            if not best_splits: self.logger.info("No nodes split."); cur.unpersist(); break 

            self.logger.debug("Creating child nodes...")
            split_map = {}; new_open_nodes_for_next_level = set()
            for pid, (measure, exemplars) in best_splits.items():
                child_dict = {}
                for branch_label in exemplars: 
                    cid = self._next_id; self._next_id += 1
                    self.tree[cid] = TreeNode(cid, pid, None, False, None, {}) 
                    child_dict[branch_label] = cid
                    split_map[(pid, branch_label)] = cid
                    new_open_nodes_for_next_level.add(cid)
                self.tree[pid] = self.tree[pid]._replace(split_on=(measure, exemplars), children=child_dict, is_leaf=False)
                self.logger.debug(f"Parent node {pid} updated. Children: {list(child_dict.values())}")

            open_nodes = new_open_nodes_for_next_level
            self.logger.debug(f"New open_nodes for next level: {open_nodes}")

            # --- P1: Use Pandas UDF for routing ---
            bc_split_map = self.spark.sparkContext.broadcast(split_map)
            bc_best_exemplars = self.spark.sparkContext.broadcast({pid: ex for pid, (_, ex) in best_splits.items()})
            self.logger.debug("Broadcasted split map and best exemplars for routing.")
            _euclid_gmm_local_route = _euclid_gmm # Local ref for UDF

            @F.pandas_udf(IntegerType())
            def route_pandas_udf(pid_series: pd.Series, ts_series: pd.Series) -> pd.Series:
                split_map_val = bc_split_map.value; exs_map_val = bc_best_exemplars.value
                results = []
                for pid, ts in zip(pid_series, ts_series):
                    if pid not in exs_map_val: results.append(pid); continue 
                    split_exemplars = exs_map_val[pid]
                    best_d, best_lbl = float("inf"), None
                    for lbl, ex_ts in split_exemplars.items():
                        d = _euclid_gmm_local_route(ts, ex_ts) 
                        if d < best_d: best_d, best_lbl = d, lbl
                    results.append(split_map_val.get((pid, best_lbl), pid))
                return pd.Series(results, dtype=pd.Int64Dtype()) # Use nullable Int

            # Apply the route UDF
            old_assign = assign 
            self.logger.info("Applying route_pandas_udf to update assignments...")
            assign = assign.withColumn("node_id", route_pandas_udf("node_id", "time_series")) \
                           .persist(StorageLevel.MEMORY_AND_DISK) # P2: Use MEMORY_AND_DISK
            assign_updated_count = assign.count() # P1: Necessary action
            self.logger.info(f"Assignment DataFrame updated. Rows: {assign_updated_count}")

            # Unpersist intermediates
            old_assign.unpersist()
            bc_split_map.unpersist(blocking=False)
            bc_best_exemplars.unpersist(blocking=False)
            cur.unpersist() 
            self.logger.debug(f"Unpersisted intermediates for depth {depth}.")

            depth += 1 
        # --- End of Tree Building Loop ---
        self.logger.info(f"Tree building loop finished at depth {depth}.")

        # --- Final Dangling Node Check --- 
        self.logger.debug("Performing final check for dangling internal nodes.")
        nodes_to_finalize = [nid for nid, nd in self.tree.items() if not nd.is_leaf and not nd.children]
        if nodes_to_finalize:
             self.logger.warning(f"Found {len(nodes_to_finalize)} dangling nodes: {nodes_to_finalize}")
             # Filter final assignment DF for these nodes
             dangling_df = assign.filter(F.col("node_id").isin(nodes_to_finalize))
             dangling_stats_rows = dangling_df.groupBy("node_id", "true_label").count().collect()
             stats_by_node = collections.defaultdict(dict)
             for r in dangling_stats_rows: stats_by_node[r["node_id"]][r["true_label"]] = r["count"]
             for nid in nodes_to_finalize:
                 stats = stats_by_node.get(nid, {}); maj_lbl = self._maj 
                 if stats: maj_lbl = max(stats.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                 self.tree[nid] = self.tree[nid]._replace(is_leaf=True, prediction=maj_lbl, split_on=None)
                 self.logger.info(f"Dangling node {nid} finalized as leaf. Prediction: {maj_lbl}.")
        
        assign.unpersist()
        self.logger.info("GlobalProxTree fitting process finished.")
        return self

    # --- P1: Prediction uses Pandas UDF ---
    def predict(self, df):
        """ Predicts class labels using Pandas UDF for traversal. """
        self.logger.info("Starting GlobalProxTree prediction (using Pandas UDF).")
        df_ts = self._to_ts_df(df) # Ensure correct format

        if not self.tree or 0 not in self.tree or (not self.tree[0].children and self.tree[0].prediction is None):
             self.logger.warning("Tree not fitted/empty. Returning default predictions.")
             default_pred = F.lit(self._maj).cast(IntegerType()).alias("prediction")
             sel_cols = ["row_id", "time_series"] + (["true_label"] if "true_label" in df_ts.columns else []) + [default_pred]
             return df_ts.select(*sel_cols)

        self.logger.debug("Converting tree to plain dict for broadcast.")
        plain_tree = {nid: node._asdict() for nid, node in self.tree.items()}
        bc_tree = self.spark.sparkContext.broadcast(plain_tree)
        self.logger.debug(f"Broadcasted plain tree ({len(plain_tree)} nodes).")

        # Need _euclid_gmm available in the UDF scope
        _euclid_gmm_local_pred = _euclid_gmm # Create local ref for UDF

        @F.pandas_udf(IntegerType())
        def traverse_pandas_udf(ts_series: pd.Series) -> pd.Series:
            tree_dict_pd = bc_tree.value # Access broadcast value once per batch
            predictions = []
            
            for ts in ts_series: # Iterate through the Pandas Series
                if ts is None: predictions.append(None); continue
                
                node_id = 0
                MAX_TRAVERSAL_DEPTH = 50; current_depth = 0
                
                while node_id in tree_dict_pd and current_depth < MAX_TRAVERSAL_DEPTH:
                    current_node = tree_dict_pd[node_id]
                    if current_node.get('is_leaf', False):
                        predictions.append(current_node.get('prediction')); break 
                    
                    split_info = current_node.get('split_on') 
                    children = current_node.get('children')
                    if not split_info or not children: predictions.append(current_node.get('prediction')); break # Fallback

                    _, exemplars = split_info 
                    if not exemplars: predictions.append(current_node.get('prediction')); break # Fallback

                    min_dist_all = float("inf"); best_branch_id_all = None 
                    for branch_id, exemplar_ts in exemplars.items():
                        d = _euclid_gmm_local_pred(ts, exemplar_ts) 
                        if d < min_dist_all: min_dist_all = d; best_branch_id_all = branch_id

                    if best_branch_id_all is not None and best_branch_id_all in children:
                        node_id = children[best_branch_id_all]
                    else: # Fallback to nearest existing child
                        min_dist_existing = float("inf"); next_node_id_found = None 
                        for ex_br_id, ex_ch_id in children.items():
                            if ex_br_id in exemplars: 
                                d = _euclid_gmm_local_pred(ts, exemplars[ex_br_id]) 
                                if d < min_dist_existing: min_dist_existing = d; next_node_id_found = ex_ch_id
                        if next_node_id_found is not None: node_id = next_node_id_found
                        else: predictions.append(current_node.get('prediction')); break # Ultimate fallback
                    
                    current_depth += 1
                else: # Handle while loop exit
                     if current_depth >= MAX_TRAVERSAL_DEPTH:
                          last_node = tree_dict_pd.get(node_id)
                          pred = last_node.get('prediction') if last_node and last_node.get('is_leaf') else None
                          predictions.append(pred)
                     else: predictions.append(None) 
                          
            return pd.Series(predictions, dtype=pd.Int64Dtype()) # Use nullable int

        self.logger.info("Applying prediction Pandas UDF...")
        out_df = df_ts.withColumn("pred_raw", traverse_pandas_udf("time_series")) \
                      .withColumn("prediction", F.coalesce(F.col("pred_raw"), F.lit(self._maj)).cast(IntegerType())) \
                      .drop("pred_raw")
        
        bc_tree.unpersist(blocking=False) 
        self.logger.debug("Unpersisted broadcasted tree.")

        # Select final output columns
        select_cols = ["row_id", "time_series"] + (["true_label"] if "true_label" in out_df.columns else []) + ["prediction"]
        return out_df.select(*select_cols)


    def print_tree(self) -> str:
        """ Returns a human-readable string representation of the tree. """
        # (Keep existing print_tree logic)
        self.logger.debug("print_tree started.")
        lines = []
        def rec(nid: int, depth: int):
             nd = self.tree.get(nid)
             if nd is None: lines.append("  " * depth + f"#{nid} MISSING"); return
             ind = "  " * depth
             if nd.is_leaf: lines.append(f"{ind}Leaf {nid} → {nd.prediction}")
             else:
                 meas, ex = nd.split_on or (None, {})
                 lines.append(f"{ind}Node {nid} split={meas} labels={list(ex.keys())}")
                 for lbl, cid in sorted(nd.children.items()):
                     lines.append(f"{ind}  ├─ lbl={lbl} → child {cid}")
                     rec(cid, depth + 2)
        rec(0, 0)
        tree_str = "\n".join(lines)
        self.logger.debug("print_tree finished.")
        return tree_str


    def save_tree(self, path: str):
        """ Pickles the essential state of the manager to a file. """
        self.logger.info(f"Saving GlobalModelManager state to {path}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True) 
            state = {
                "max_depth": self.max_depth, "min_samples": self.min_samples,
                "k": self.k, "tree": self.tree, 
                "_next_id": self._next_id, "_maj": self._maj,
                "random_state": self.random_state # P3: Save seed
            }
            with open(path, "wb") as fh: pickle.dump(state, fh)
            self.logger.info(f"Successfully saved state to {path}.")
        except Exception as e: self.logger.error(f"Failed to save tree state: {e}", exc_info=True)


    @classmethod
    def load_tree(cls, spark: SparkSession, path: str) -> "GlobalModelManager":
        """ Loads the manager state from a pickled file. """
        logger_gmm.info(f"Loading GlobalModelManager state from {path}") # Use class logger
        try:
            with open(path, "rb") as fh: state: Dict[str, Any] = pickle.load(fh)
            logger_gmm.debug("State loaded successfully.")
        except Exception as e: logger_gmm.error(f"Failed to load tree state: {e}", exc_info=True); raise

        # Reconstruct config for initialization
        loaded_config = {
            "tree_params": {
                "max_depth": state.get("max_depth"), 
                "min_samples_split": state.get("min_samples", 2), 
                "n_splitters": state.get("k", 5), 
                "random_state": state.get("random_state") # P3: Load seed
            }
        }
        logger_gmm.debug(f"Reconstructed config: {loaded_config}")

        # Create instance and restore state
        inst = cls(spark, loaded_config)
        inst.tree = state.get("tree", {0: TreeNode(0, None, None, False, None, {})}) 
        inst._next_id = state.get("_next_id", 1) 
        inst._maj = state.get("_maj", 1) 
        # P3: Re-initialize RNG if state loaded
        inst._rng = random.Random(inst.random_state) 
        logger_gmm.info(f"Instance created. Tree size: {len(inst.tree)} nodes.")
        return inst

