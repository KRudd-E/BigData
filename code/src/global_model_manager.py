# Optimised GlobalModelManager – DF-only version (avoids large RDD shuffle)
# Enhanced Prediction Traversal
# --------------------------------------------------------------------------
# • single-pass aggregations & fewer .count() actions (RETAINED)
# • map-side combiners (reduceByKey) replaced with DataFrame ops (RETAINED)
# • strategic cache / unpersist (RETAINED)
# • NumPy-accelerated distance (RETAINED)
#   • Enhanced prediction traversal logic (IMPORTED from GlobalProxTree)
#   • Explicitly broadcast plain tree dict for prediction (GOOD PRACTICE)
#   • Refined local leaf prediction calculation in fit
#   • Corrected random exemplar sampling in fit
#   • Using Python's logging module for debug messages (NOW CONSISTENTLY)
#   • Removed explicit print statements for debugging visibility
#   • FIXED: AttributeError 'c' when accessing count
# • **public API**: fit(), predict(), save_tree(), load_tree(), print_tree()
# --------------------------------------------------------------------------

from __future__ import annotations

from pyspark.sql import SparkSession, Row # Import Row
import pyspark.sql.functions as F
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
    ArrayType,
    LongType,
)
import collections, random, math, pickle, os
from typing import Dict, Any, Tuple
import logging # Import the logging module
import sys # Import sys

# Configure logging for this module
# Note: In a distributed environment, the root logger config might be set by Spark.
# However, getting a module-specific logger like below is standard practice.
# The level set here acts as a minimum threshold for this logger.
# The actual output depends on the *handler* configuration (e.g., basicConfig)
# and the overall Spark/root logger level.
# We keep basicConfig for potential use when running as a standalone script __main__.
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress excessive logging from py4j and pyspark itself
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)


try:
    import numpy as np

    _NP = True
except ImportError: # pragma: no cover – NumPy optional for tiny envs
    _NP = False

# .............................................................................
# helpers
# .............................................................................

TreeNode = collections.namedtuple(
    "TreeNode", "node_id parent_id split_on is_leaf prediction children".split()
)

# Keep the original efficient euclidean distance function
def _euclid(a, b):
    """Fast Euclidean distance for python *or* NumPy inputs."""
    # --- Use logger for debugging ---
    # Note: Logs from UDFs go to worker logs. In local mode, often appear in console.
    logger.debug(f"UDF: _euclid inputs: a={a[:5] if isinstance(a, (list, np.ndarray)) else a}..., b={b[:5] if isinstance(b, (list, np.ndarray)) else b}...")
    # -------------------------------------
    if a is None or b is None or len(a) != len(b):
        # --- Use logger ---
        logger.debug(f"UDF: _euclid returning inf due to None/len mismatch: a is None={a is None}, b is None={b is None}, len(a)={len(a) if a is not None else 'N/A'}, len(b)={len(b) if b is not None else 'N/A'}")
        # --------------------
        return float("inf")
    if _NP:
        try:
            diff = np.subtract(a, b, dtype=float)
            dist = float(np.sqrt(np.dot(diff, diff)))
            # --- Use logger ---
            # logger.debug(f"UDF: _euclid (NumPy) returning {dist}") # Avoid logging too much if successful
            # --------------------
            return dist
        except Exception as e:
            # --- Use logger for NumPy errors ---
            logger.error(f"UDF: ERROR in _euclid (NumPy path): {e}. Inputs: a={a[:5] if isinstance(a, (list, np.ndarray)) else a}..., b={b[:5] if isinstance(b, (list, np.ndarray)) else b}...")
            # ---------------------------------------
            # Re-raise or return inf, depending on desired behavior on error
            return float("inf") # Or raise e
    else: # Pure Python path
        try:
            dist = float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
            # --- Use logger ---
            # logger.debug(f"UDF: _euclid (Python) returning {dist}") # Avoid logging too much if successful
            # --------------------
            return dist
        except Exception as e:
            # --- Use logger for Python errors ---
            logger.error(f"UDF: ERROR in _euclid (Python path): {e}. Inputs: a={a[:5] if isinstance(a, (list, np.ndarray)) else a}..., b={b[:5] if isinstance(b, (list, np.ndarray)) else b}...")
            # --------------------------------------
            # Re-raise or return inf
            return float("inf") # Or raise e


# .............................................................................
# prediction-side helper – pure python, broadcasted once per executor
# (MODIFIED to use enhanced traversal logic)
# .............................................................................

def _enhanced_mk_traverse(bc_plain_tree):
    """
    Return a local function that navigates the broadcast tree using enhanced logic.
    Expects a broadcasted plain dictionary structure.
    """

    # The tree is now a plain dictionary
    tree: Dict[int, Dict[str, Any]] = bc_plain_tree.value
    # Get a logger instance inside the UDF factory function
    # This logger will be serialized and sent to workers
    udf_logger = logging.getLogger(__name__)


    def _enhanced_traverse(ts):
        """Enhanced traversal logic for a single time series."""
        # Logging inside UDFs can be tricky; messages go to worker logs by default.
        # Use sparingly or configure Spark logging to collect worker logs.
        # udf_logger.debug("UDF: _enhanced_traverse started for time series.")

        if ts is None:
            # udf_logger.debug("UDF: Input time series is None. Returning None.")
            # Fallback handled by coalesce in predict method
            return None

        node_id = 0 # Start at root
        # udf_logger.debug(f"UDF: Starting traversal from root node {node_id}.")

        # Traverse the tree until a leaf node is reached or traversal stops
        while node_id in tree:
            current_node = tree[node_id]
            # udf_logger.debug(f"UDF: Current node_id: {node_id}, is_leaf: {current_node.get('is_leaf', False)}")

            # If it's a leaf node, return its prediction
            if current_node.get('is_leaf', False): # Use .get for safety
                # udf_logger.debug(f"UDF: Node {node_id} is leaf. Returning prediction: {current_node.get('prediction')}")
                return current_node.get('prediction') # Prediction is in the plain dict

            # If it's an internal node, use the split info to decide which branch to follow
            split_info = current_node.get('split_on') # (measure_type, {branch_id: exemplar_ts})
            children = current_node.get('children')

            # Ensure split info and children exist for internal nodes
            if split_info and children and len(children) > 0:
                _, exemplars = split_info # We only need exemplars for the split
                # udf_logger.debug(f"UDF: Node {node_id} is internal. Split info: {split_info}, Children: {children}")

                # Calculate distance to ALL exemplars used in THIS node's split
                min_dist_all_exemplars = float("inf")
                best_branch_id_all_exemplars = None # Label of the nearest exemplar

                # Handle case where exemplars might be empty (shouldn't happen in valid tree)
                if not exemplars:
                    # udf_logger.warning(f"UDF: Node {node_id} is internal but has no exemplars. Treating as leaf.")
                    # Internal node with no exemplars or children? Treat as leaf with its prediction
                    return current_node.get('prediction') # Prediction should be None if not finalized

                # udf_logger.debug(f"UDF: Calculating distances to exemplars for node {node_id}.")
                for branch_id, exemplar_ts in exemplars.items():
                    # Use the base _euclid function
                    d = _euclid(ts, exemplar_ts)
                    # udf_logger.debug(f"UDF: Distance to exemplar {branch_id}: {d}")
                    if d < min_dist_all_exemplars:
                        min_dist_all_exemplars = d
                        best_branch_id_all_exemplars = branch_id

                # udf_logger.debug(f"UDF: Nearest exemplar branch_id for node {node_id}: {best_branch_id_all_exemplars}")

                # --- Enhanced Traversal Logic ---
                # Check if the child node corresponding to the overall nearest exemplar exists
                if best_branch_id_all_exemplars is not None and best_branch_id_all_exemplars in children:
                    # If the child exists, move to that child node
                    next_node = children[best_branch_id_all_exemplars]
                    # udf_logger.debug(f"UDF: Moving to child node {next_node} via branch {best_branch_id_all_exemplars}.")
                    node_id = next_node
                else:
                    # If the ideal child does NOT exist (pruned branch),
                    # find the nearest exemplar among the *existing* child branches and follow that path.
                    # udf_logger.debug(f"UDF: Ideal child branch {best_branch_id_all_exemplars} not found in children {children}. Finding nearest among existing children.")
                    min_dist_existing_children = float("inf")
                    next_node_id = None # The child node ID to move to

                    # Iterate through the *existing* child branches listed in the tree structure
                    for existing_branch_id, existing_child_id in children.items():
                        # Find the exemplar time series for this existing branch from the original exemplars used for the split
                        if existing_branch_id in exemplars: # Double check exemplar exists for this branch
                            existing_exemplar_ts = exemplars[existing_branch_id]
                            # Calculate distance to this existing branch's exemplar
                            d = _euclid(ts, existing_exemplar_ts)
                            # udf_logger.debug(f"UDF: Distance to existing child branch {existing_branch_id} exemplar: {d}")
                            if d < min_dist_existing_children:
                                min_dist_existing_children = d
                                next_node_id = existing_child_id

                    # If a nearest existing child was found, move to that child node
                    if next_node_id is not None:
                        # udf_logger.debug(f"UDF: Routing to nearest existing child {next_node_id}.")
                        node_id = next_node_id
                    else:
                        # If no existing children were found or no nearest existing child determined,
                        # stop traversal and return the current node's prediction (which should be None for internal nodes)
                        # udf_logger.warning(f"UDF: No nearest existing child found for node {current_node.get('node_id')}. Stopping traversal.")
                        return current_node.get('prediction') # Fallback handled by coalesce later


            else:
                # If the node is internal but has no split info or children (shouldn't happen in valid tree)
                # Stop traversal and return the current node's prediction.
                # udf_logger.warning(f"UDF: Node {current_node.get('node_id')} is internal but missing split info or children. Stopping traversal.")
                return current_node.get('prediction') # Fallback handled by coalesce later


        # If the loop finishes without returning (e.g., node_id not found, error)
        # This indicates a problem in the tree structure.
        # Returning None here, will be caught by coalesce.
        # udf_logger.error(f"UDF: Traversal loop finished unexpectedly at node_id {node_id}.")
        return None


    return _enhanced_traverse


# =============================================================================
# GlobalModelManager class (Optimised + Enhanced Prediction)
# =============================================================================

class GlobalModelManager:
    """Distribution-friendly proximity-tree learner."""

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        logger.debug("GlobalModelManager __init__ started.")
        p = config["tree_params"]
        self.spark = spark
        self.max_depth: int | None = p.get("max_depth") # Use .get for safety
        self.min_samples: int = p.get("min_samples_split", 2) # Use .get with default
        self.k: int = p.get("n_splitters", 5) # Use .get with default
        self.tree: Dict[int, TreeNode] = {0: TreeNode(0, None, None, False, None, {})}
        self._next_id: int = 1
        self._maj: int = 1 # fallback class if everything else fails
        logger.debug(f"Initialized with max_depth={self.max_depth}, min_samples={self.min_samples}, k={self.k}")
        logger.debug("GlobalModelManager __init__ finished.")


    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _to_ts_df(self, df):
        """Ensure DataFrame has (row_id, time_series[, true_label])."""
        logger.debug("_to_ts_df started.")

        if "row_id" not in df.columns:
            logger.debug("Adding row_id.")
            df = df.withColumn("row_id", F.monotonically_increasing_id())
        else:
            logger.debug("Casting existing row_id to LongType.")
            df = df.withColumn("row_id", F.col("row_id").cast(LongType()))

        if "time_series" in df.columns:
            logger.debug("'time_series' column already exists.")
            if "label" in df.columns and "true_label" not in df.columns:
                logger.debug("Renaming 'label' to 'true_label'.")
                df = df.withColumnRenamed("label", "true_label")
            logger.debug("_to_ts_df finished (already formatted).")
            return df

        lbl = "label" if "label" in df.columns else (
            "true_label" if "true_label" in df.columns else None
        )
        feat_cols = [c for c in df.columns if c not in {lbl, "row_id"}]
        logger.debug(f"Found feature columns: {feat_cols}")
        cols = [
            "row_id",
            F.array(*[F.col(c) for c in feat_cols]).alias("time_series"),
        ]
        if lbl:
            logger.debug(f"Including label column: {lbl}")
            cols.append(F.col(lbl).cast(IntegerType()).alias("true_label"))
        else:
            logger.debug("No label column found.")

        ts_df = df.select(*cols)
        logger.debug("_to_ts_df finished (conversion done).")
        return ts_df

    @staticmethod
    def _gini(counts: Dict[int, int]) -> float:
        tot = sum(counts.values())
        if tot == 0:
            return 0.0
        return 1.0 - sum((c / tot) ** 2 for c in counts.values())

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, df): # noqa: C901 (complexity accepted here)
        """Train the proximity tree."""
        logger.debug("fit started.")
        df = self._to_ts_df(df).cache()
        initial_row_count = df.count()
        logger.debug(f"Initial DataFrame prepared and cached. Row count: {initial_row_count}")

        if initial_row_count == 0:
             logger.debug("Input DataFrame is empty. Setting root as leaf.")
             self.tree[0] = self.tree[0]._replace(is_leaf=True, prediction=self._maj, children={})
             df.unpersist()
             logger.debug("fit finished (empty DataFrame).")
             return self


        maj_row = df.groupBy("true_label").count().orderBy(F.desc("count")).first()
        if maj_row:
            self._maj = maj_row[0]
            logger.debug(f"Overall majority class calculated: {self._maj}")
        else:
            logger.debug(f"No data to calculate overall majority class. Keeping default: {self._maj}")


        assign = (
            df.select("row_id", "time_series", "true_label")
            .withColumn("node_id", F.lit(0))
            .cache()
        )
        assign_initial_count = assign.count()
        logger.debug(f"Initial assignment DataFrame created and cached. Row count: {assign_initial_count} at root node 0.")
        df.unpersist() # Unpersist initial DataFrame


        open_nodes, depth = {0}, 0
        logger.debug(f"Starting tree building loop with initial open_nodes: {open_nodes}")

        # --------------- depth-wise growth ----------------------------
        while open_nodes and (self.max_depth is None or depth < self.max_depth):
            logger.debug(f"\n--- Starting tree level {depth} ---")
            logger.debug(f"Open nodes for this level: {open_nodes}")

            cur = assign.filter(F.col("node_id").isin(list(open_nodes))).cache()
            cur_count = cur.count()
            logger.debug(f"Filtered data for current level. Row count: {cur_count}")

            if cur.isEmpty():
                logger.debug(f"No data for open nodes at depth {depth}. Breaking loop.")
                cur.unpersist(); break

            # 1) exemplar pool per (node,label) – single pass
            logger.debug("Starting exemplar pool sampling.")
            all_ts_per_node_label = (
                cur.groupBy("node_id", "true_label")
                .agg(F.collect_list("time_series").alias("ts_list"))
                .collect() # Collects list of Rows: Row(node_id=..., true_label=..., ts_list=[...])
            )
            logger.debug(f"Collected time series lists for {len(all_ts_per_node_label)} node-label groups.")

            pool: Dict[int, Dict[int, list]] = {}
            # Perform random sampling on the driver from the collected lists
            for r in all_ts_per_node_label:
                node_id = r.node_id
                true_label = r.true_label
                ts_list = r.ts_list

                if not ts_list:
                    logger.debug(f"No time series found for node {node_id}, label {true_label} in collected list.")
                    continue

                # Randomly sample self.k exemplars from the list
                # Ensure we don't sample more than available
                num_to_sample = min(self.k, len(ts_list))
                # Use random.sample for actual random selection
                sampled_ts = random.sample(ts_list, num_to_sample)

                pool.setdefault(node_id, {})[true_label] = sampled_ts
                # logger.debug(f"Sampled {len(sampled_ts)} exemplars for node {node_id}, label {true_label}. Sample: {sampled_ts[:2]}...") # Avoid printing large lists
                logger.debug(f"Sampled {len(sampled_ts)} exemplars for node {node_id}, label {true_label}.")


            logger.debug(f"Finished exemplar pool sampling. Pool structure keys: {list(pool.keys())}")


            best: Dict[int, Tuple[str, Dict[int, list]]] = {}
            to_leaf: set[int] = set()

            # 2) pick best split per node (driver)
            logger.debug("Starting best split evaluation per node.")
            for nid in list(open_nodes): # Iterate over a copy in case nodes are removed from open_nodes
                logger.debug(f"Evaluating splits for node {nid}.")
                nd_df = cur.filter(F.col("node_id") == nid)
                nd_df_count = nd_df.count()
                logger.debug(f"Data count for node {nid}: {nd_df_count}")

                # Calculate local stats for leaf conditions and parent Gini
                stats_rows = nd_df.groupBy("true_label").count().collect()
                # --- FIXED: Access count using r['count'] ---
                stats = {r.true_label: r['count'] for r in stats_rows}
                tot = sum(stats.values())
                logger.debug(f"Node {nid} stats: {stats}, total samples: {tot}")


                # Leaf Condition 1: Insufficient samples, purity, or no exemplars
                if tot < self.min_samples:
                    logger.debug(f"Node {nid} has {tot} samples, below min_samples {self.min_samples}. Marking as leaf.")
                    to_leaf.add(nid); continue
                if len(stats) <= 1:
                     logger.debug(f"Node {nid} is pure ({len(stats)} labels). Marking as leaf.")
                     to_leaf.add(nid); continue
                if nid not in pool or not pool[nid]:
                    logger.debug(f"No exemplars found in pool for node {nid}. Marking as leaf.")
                    to_leaf.add(nid); continue

                parent_g = self._gini(stats)
                logger.debug(f"Node {nid} parent Gini: {parent_g}")

                labels = list(pool[nid].keys())

                # Leaf Condition 2: Insufficient exemplar labels for split
                if len(labels) < 2:
                    logger.debug(f"Node {nid} has {len(labels)} exemplar labels in pool, need >= 2 for split. Marking as leaf.")
                    to_leaf.add(nid); continue

                best_gain, best_exp = -1.0, None
                logger.debug(f"Evaluating {self.k} candidate splits for node {nid}.")
                for i in range(self.k): # Evaluate k candidate splits
                    # Sample exemplars for THIS candidate split from the pool
                    # Ensure pool[nid][lbl] is not empty before random.choice
                    candidate_ex = {}
                    for lbl in labels:
                        if lbl in pool[nid] and pool[nid][lbl]:
                            candidate_ex[lbl] = random.choice(pool[nid][lbl])
                        else:
                            logger.debug(f"Warning: No exemplars in pool for label {lbl} in node {nid} for candidate {i+1}. Skipping this candidate split.")
                            candidate_ex = None # Invalidate this candidate
                            break # Stop evaluating this candidate

                    if candidate_ex is None or len(candidate_ex) < 2:
                         logger.debug(f"Candidate split {i+1} for node {nid} has less than 2 exemplars ({len(candidate_ex) if candidate_ex is not None else 'None'}). Skipping.")
                         continue # Need at least two branches for a valid split

                    # logger.debug(f"Evaluating candidate split {i+1} for node {nid} with exemplars for labels: {list(candidate_ex.keys())}. Exemplars: {candidate_ex}")
                    logger.debug(f"Evaluating candidate split {i+1} for node {nid} with exemplars for labels: {list(candidate_ex.keys())}.")
                    bc_ex = self.spark.sparkContext.broadcast(candidate_ex)

                    @F.udf(IntegerType())
                    def nearest_lbl_udf(ts):
                        # NOTE: Logs from UDFs go to worker logs!
                        # Use logger here instead of print
                        udf_logger_local = logging.getLogger(__name__) # Get logger instance in worker
                        udf_logger_local.debug(f"UDF: nearest_lbl_udf processing TS: {ts[:5] if isinstance(ts, (list, np.ndarray)) else ts}...")

                        best_d, best_l = float("inf"), None
                        # Use the original _euclid function
                        exemplars_val = bc_ex.value
                        if not exemplars_val:
                            udf_logger_local.debug("UDF: Exemplars is empty, returning None.")
                            return None # Safety check

                        for l, ex_ts in exemplars_val.items():
                            # Use logger inside _euclid as well
                            d = _euclid(ts, ex_ts) # _euclid now uses logger internally
                            if d < best_d:
                                best_d, best_l = d, l

                        udf_logger_local.debug(f"UDF: Finished calculating distances. Best label: {best_l}")
                        return best_l

                    # This is the DataFrame-based Gini calculation (RETAINED for speed)
                    ass = (
                        nd_df.withColumn("branch", nearest_lbl_udf("time_series"))
                        .groupBy("branch", "true_label")
                        .count()
                    )
                    branch_cnt = ass.groupBy("branch").agg(F.sum("count").alias("tot"))
                    joined = ass.join(branch_cnt, "branch")

                    # Check if joined DataFrame is empty before calculating impurity
                    if joined.isEmpty():
                         logger.debug(f"Joined DataFrame is empty for candidate {i+1} on node {nid}. Cannot calculate impurity.")
                         bc_ex.unpersist(False)
                         continue

                    imp_row = (
                         joined.withColumn("prob_sq", (F.col("count") / F.col("tot")) ** 2)
                         .groupBy("branch", "tot")
                         .agg(F.sum("prob_sq").alias("s"))
                         .withColumn("g", 1.0 - F.col("s"))
                         .withColumn("w", (F.col("tot") / tot) * F.col("g"))
                         .agg(F.sum("w").alias("imp"))
                         .first() # Collect the single result to the driver
                    )

                    if imp_row is None:
                         logger.debug(f"Impurity calculation returned None for candidate {i+1} on node {nid}. Skipping.")
                         bc_ex.unpersist(False)
                         continue

                    imp = imp_row[0]
                    gain = parent_g - imp
                    logger.debug(f"Candidate split {i+1} for node {nid}: Impurity={imp:.4f}, Gain={gain:.4f}")

                    bc_ex.unpersist(False) # Unpersist broadcasted exemplars for this candidate

                    # Leaf Condition 3: Update best split if gain is improved
                    if gain > best_gain:
                        best_gain, best_exp = gain, candidate_ex
                        logger.debug(f"Candidate split {i+1} is the best so far for node {nid} with gain {best_gain:.4f}.")

                # Leaf Condition 4: If best gain is not significantly positive after all candidates
                if best_gain > 1e-9: # Use tolerance for splitting decision
                    logger.debug(f"Node {nid} found a good split with gain {best_gain:.4f}.")
                    best[nid] = ("euclidean", best_exp)
                else:
                    logger.debug(f"Node {nid} did not find a good split (best gain {best_gain:.4f}). Marking as leaf.")
                    to_leaf.add(nid) # Node becomes a leaf

            logger.debug("Finished best split evaluation per node.")


            # 2b) mark leaves right away **with LOCAL majority** (REFINED CALCULATION)
            logger.debug("Finalizing nodes marked as leaves in this iteration.")
            for nid in list(to_leaf): # Iterate over a copy as we remove from open_nodes
                if nid not in self.tree:
                     logger.debug(f"Node {nid} already removed from tree? Skipping finalization.")
                     open_nodes.discard(nid) # Ensure it's not in open_nodes
                     continue

                if self.tree[nid].is_leaf:
                     logger.debug(f"Node {nid} already finalized as leaf. Skipping.")
                     open_nodes.discard(nid) # Ensure it's not in open_nodes
                     continue

                logger.debug(f"Finalizing node {nid} as a leaf.")
                # Recalculate stats from the data currently assigned to this node (more accurate)
                leaf_data_df = cur.filter(F.col("node_id") == nid).cache()
                leaf_stats_rows = leaf_data_df.groupBy("true_label").count().collect()
                # --- FIXED: Access count using r['count'] ---
                leaf_stats = {r.true_label: r['count'] for r in leaf_stats_rows}
                leaf_data_df.unpersist() # Unpersist leaf data

                logger.debug(f"Node {nid} local stats for leaf prediction: {leaf_stats}")

                maj_lbl = self._maj # Default to overall majority

                if leaf_stats:
                    # Find majority label using count, break ties using smallest label
                    maj_lbl = max(leaf_stats.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                    logger.debug(f"Node {nid} local majority prediction: {maj_lbl}")
                else:
                    logger.debug(f"No data found for node {nid} during leaf finalization. Using overall majority fallback: {maj_lbl}")


                self.tree[nid] = self.tree[nid]._replace(is_leaf=True, prediction=maj_lbl, children={})
                logger.debug(f"Node {nid} marked as leaf with prediction {maj_lbl}.")
                open_nodes.discard(nid) # Remove finalized leaves from consideration

            logger.debug("Finished finalizing leaves for this iteration.")


            # 3) create children + update assignment DF
            # This block is largely REETAINED from the original for its DataFrame efficiency
            if not best:
                logger.debug("No nodes found good splits in this iteration. Breaking loop.")
                cur.unpersist(); break # No nodes successfully split in this iteration

            logger.debug("Creating children and updating assignment DataFrame.")
            split_map, new_open = {}, {}
            for pid, (m, ex) in best.items():
                logger.debug(f"Processing best split for parent node {pid}.")
                ch = {}
                # Only create children for branches with exemplars in the chosen split
                for lbl in ex:
                    cid = self._next_id; self._next_id += 1
                    # Children are initially non-leaves with no prediction or split info
                    self.tree[cid] = TreeNode(cid, pid, None, False, None, {})
                    ch[lbl] = cid
                    # Map (parent_id, branch_label) to new child_id
                    split_map[(pid, lbl)] = cid
                    # Add new children to the set for the next iteration
                    new_open[cid] = None # Value doesn't matter, just need the set keys
                    logger.debug(f"Created child {cid} for branch {lbl} of parent {pid}.")


                # Update the parent node in the tree structure
                self.tree[pid] = self.tree[pid]._replace(split_on=(m, ex), children=ch, is_leaf=False)
                logger.debug(f"Parent node {pid} updated: split_on={self.tree[pid].split_on}, children={self.tree[pid].children}")


            # Prepare for the next iteration
            open_nodes = set(new_open.keys())
            logger.debug(f"New open_nodes for next level: {open_nodes}")

            # Broadcast the split mapping and exemplars for the route UDF
            bc_split = self.spark.sparkContext.broadcast(split_map)
            bc_exs = self.spark.sparkContext.broadcast({pid: ex for pid, (_, ex) in best.items()})
            logger.debug("Broadcasted split_map and best_exs for route_udf.")


            # Define the UDF for routing rows to children (RETAINED)
            @F.udf(IntegerType())
            def route_udf(pid, ts):
                # If this parent didn't split in this iteration (shouldn't happen if filtering 'cur' correctly)
                if pid not in bc_exs.value:
                    # This case might happen if a node was in open_nodes but had no data in cur
                    # or if there's a logic error. Returning pid keeps the row at the current node.
                    return pid

                ex = bc_exs.value[pid]
                best_d, best_lbl = float("inf"), None
                # Find nearest exemplar for the row among the split exemplars
                for l, ex_ts in ex.items():
                    d = _euclid(ts, ex_ts)
                    if d < best_d:
                        best_d, best_lbl = d, l

                # Get the new child node ID from the split mapping.
                # If the (parent_id, branch_label) is NOT in the map (e.g., branch didn't meet min_samples)
                # return the parent_id, effectively keeping the row at the parent node.
                return bc_split.value.get((pid, best_lbl), pid)

            # Apply the route UDF to the entire assignment DataFrame
            # This updates the node_id for all rows that were in splitting nodes
            old_assign = assign # Keep reference to unpersist
            logger.debug("Applying route_udf to update assign DataFrame.")
            assign = assign.withColumn("node_id", route_udf("node_id", "time_series")).cache()
            assign_updated_count = assign.count() # Trigger action and cache
            logger.debug(f"assign DataFrame updated and cached. New total rows: {assign_updated_count}")


            # Unpersist previous assignment DF and broadcasts
            old_assign.unpersist()
            bc_split.unpersist(False)
            bc_exs.unpersist(False)
            cur.unpersist() # Unpersist the current level's data
            logger.debug(f"Unpersisted intermediates for depth {depth}.")

            depth += 1 # Increment depth for next iteration

        logger.debug("\n--- Main tree building loop finished ---")
        logger.debug(f"Final open_nodes: {open_nodes}")


        # --- Final Dangling Node Finalization (Using the final 'assign' state) ---
        # This block should execute *before* the final assign.unpersist()
        logger.debug("Performing final dangling node finalization.")
        nodes_to_finalize_at_end = [nid for nid, nd in self.tree.items() if not nd.is_leaf and not nd.children]
        if nodes_to_finalize_at_end:
             logger.debug(f"Found {len(nodes_to_finalize_at_end)} dangling nodes to finalize.")
             # Filter the final assignment DF for these dangling nodes
             dangling_df = assign.filter(F.col("node_id").isin(nodes_to_finalize_at_end)).cache()
             dangling_df_count = dangling_df.count()
             logger.debug(f"Data count for dangling nodes: {dangling_df_count}")


             # Calculate local majority for each dangling node
             dangling_stats_rows = dangling_df.groupBy("node_id", "true_label").count().collect()
             dangling_stats_by_node = collections.defaultdict(dict)
             for r in dangling_stats_rows:
                 # --- FIXED: Access count using r['count'] ---
                 dangling_stats_by_node[r.node_id][r.true_label] = r['count']

             dangling_df.unpersist() # Unpersist dangling data
             logger.debug("Dangling DataFrame unpersisted.")


             for nid in nodes_to_finalize_at_end:
                 stats = dangling_stats_by_node.get(nid, {})
                 maj_lbl = self._maj # fallback

                 if stats:
                     maj_lbl = max(stats.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                     logger.debug(f"Dangling node {nid} local majority prediction: {maj_lbl}")
                 else:
                     logger.debug(f"No data found for dangling node {nid}. Using overall majority fallback: {maj_lbl}")


                 self.tree[nid] = self.tree[nid]._replace(is_leaf=True, prediction=maj_lbl, split_on=None)
                 logger.debug(f"Dangling node {nid} finalized as leaf with prediction {maj_lbl}.")
        else:
             logger.debug("No dangling nodes found to finalize at the end.")


        # Now unpersist the final assign DataFrame
        assign.unpersist()
        logger.debug("Final assign DataFrame unpersisted.")


        logger.debug("fit finished.")
        return self

    # ------------------------------------------------------------------
    # prediction (MODIFIED for enhanced traversal and plain dict broadcast)
    # ------------------------------------------------------------------

    def predict(self, df):
        logger.debug("predict started.")
        df = self._to_ts_df(df)
        df_count = df.count()
        logger.debug(f"Input DataFrame for prediction prepared. Row count: {df_count}")


        if not self.tree or (0 not in self.tree) or (self.tree[0].prediction is None and not self.tree[0].children):
             logger.warning("Tree is not fitted or is empty. Returning DataFrame with default prediction.")
             # Return DataFrame with default prediction if tree is not usable
             default_pred_col = F.lit(self._maj).cast(IntegerType()).alias("prediction")
             select_cols = ["row_id", "time_series"] + (["true_label"] if "true_label" in df.columns else []) + [default_pred_col]
             return df.select(*select_cols)


        # --- Convert tree structure to a plain dictionary for broadcasting ---
        logger.debug("Converting tree structure to plain dictionary for broadcasting.")
        plain_tree_structure = {}
        for node_id, node in self.tree.items():
            plain_tree_structure[node_id] = {
                'node_id': node.node_id,
                'parent_id': node.parent_id,
                # Ensure split_on is a plain structure (tuple of string and dict)
                'split_on': node.split_on,
                'is_leaf': node.is_leaf,
                # Ensure prediction is serializable (should be int or None)
                'prediction': node.prediction, # Should be int or None already
                # Children dictionary keys (branch_id) and values (child_node_id) are already plain types
                'children': node.children
            }
        logger.debug(f"Converted tree structure to plain dictionary with {len(plain_tree_structure)} nodes.")


        # Broadcast the plain tree structure
        logger.debug("Broadcasting plain tree structure.")
        bc_tree = self.spark.sparkContext.broadcast(plain_tree_structure)
        logger.debug("Broadcasted plain tree structure.")


        # Use the enhanced traversal function
        udf_pred = F.udf(_enhanced_mk_traverse(bc_tree), IntegerType())
        logger.debug("Created prediction UDF.")

        # Apply UDF and coalesce with overall majority as a final fallback
        logger.debug("Applying prediction UDF and coalescing results.")
        out = (
            df.withColumn("pred", udf_pred("time_series"))
            .withColumn("prediction", F.coalesce("pred", F.lit(self._maj)))
            .drop("pred")
        )
        out_count = out.count() # Trigger action
        logger.debug(f"Prediction applied. Output DataFrame row count: {out_count}")


        bc_tree.unpersist(False)
        logger.debug("Broadcasted tree unpersisted.")

        # Select relevant output columns
        sel = ["row_id", "time_series"] + (["true_label"] if "true_label" in out.columns else []) + ["prediction"]
        logger.debug(f"Selecting final columns: {sel}")
        final_output_df = out.select(*sel)

        logger.debug("predict finished.")
        return final_output_df

    # ------------------------------------------------------------------
    # utilities – print / save / load (RETAINED)
    # ------------------------------------------------------------------

    def print_tree(self) -> str:
        """Return a human-readable representation (driver-side)."""
        logger.debug("print_tree started.")
        lines = []

        def rec(nid: int, depth: int):
            nd = self.tree.get(nid)
            if nd is None:
                lines.append("  " * depth + f"#{nid} MISSING")
                return
            ind = "  " * depth
            if nd.is_leaf:
                lines.append(f"{ind}Leaf {nid} → {nd.prediction}")
            else:
                meas, ex = nd.split_on or (None, {})
                lines.append(f"{ind}Node {nid} split={meas} labels={list(ex.keys())}")
                for lbl, cid in sorted(nd.children.items()):
                    lines.append(f"{ind}  ├─ lbl={lbl} → child {cid}")
                    rec(cid, depth + 2)

        rec(0, 0)
        tree_str = "\n".join(lines)
        logger.debug("print_tree finished.")
        return tree_str

    # .................................................................
    # persistence helpers (RETAINED)
    # .................................................................

    def save_tree(self, path: str):
        """Pickle the *entire* manager (tree + params) to a file."""
        logger.debug(f"save_tree started. Path: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "max_depth": self.max_depth,
                "min_samples": self.min_samples,
                "k": self.k,
                "tree": self.tree, # Tree contains TreeNode namedtuples
                "_next_id": self._next_id,
                "_maj": self._maj,
            }, fh)
        logger.debug(f"Tree saved successfully to {path}.")
        logger.debug("save_tree finished.")


    @classmethod
    def load_tree(cls, spark: SparkSession, path: str) -> "GlobalModelManager":
        logger.debug(f"load_tree started. Path: {path}")
        try:
            with open(path, "rb") as fh:
                data: Dict[str, Any] = pickle.load(fh)
            logger.debug("Data loaded successfully from pickle.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load tree from {path}: {e}")
            raise

        dummy_conf = {
            "tree_params": {
                "max_depth": data.get("max_depth"), # Use .get for safety
                "min_samples_split": data.get("min_samples", 2), # Use .get with default
                "n_splitters": data.get("k", 5), # Use .get with default
            }
        }
        logger.debug(f"Loaded hyperparameters: {dummy_conf['tree_params']}")

        inst = cls(spark, dummy_conf)
        inst.tree = data.get("tree", {}) # Use .get with default
        inst._next_id = data.get("_next_id", 1) # Use .get with default
        inst._maj = data.get("_maj", 1) # Use .get with default
        logger.debug(f"Instance created and state restored. Root node exists: {0 in inst.tree}")
        logger.debug("load_tree finished.")
        return inst
