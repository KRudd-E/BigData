import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, ArrayType, MapType
import random
import collections
import math # For Euclidean distance
import json # To potentially serialize complex split_on info if needed, though plain dict is better
import traceback # Import traceback to print error details
import pickle # For saving/loading the tree object
import os # For path manipulation

# Define TreeNode namedtuple at the module level
TreeNode = collections.namedtuple(
    "TreeNode",
    "node_id parent_id split_on is_leaf prediction children".split()
)

# Define a simple Euclidean distance function for use in UDF
# In a real implementation, this would handle multiple distance measures and parameters
def euclidean_distance(ts1, ts2):
    """Calculates Euclidean distance between two time series."""
    if ts1 is None or ts2 is None or len(ts1) != len(ts2):
        return float('inf') # Handle invalid inputs
    # Ensure both are lists of numbers
    try:
        dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(ts1, ts2)]))
        return float(dist) # Return as float
    except Exception as e:
        # Print error only in debug mode or with proper logging
        # print(f"Error calculating distance: {e}")
        return float('inf')


# Define a UDF for predicting a single time series instance
# This UDF will need access to the tree structure (broadcasted plain dictionary)
def predict_udf_func(plain_tree_structure_broadcast):
    """
    Returns a UDF that traverses the tree for a single time series instance.
    plain_tree_structure_broadcast: Broadcast variable containing the plain dictionary tree structure.
    """
    tree = plain_tree_structure_broadcast.value

    def traverse_tree(time_series):
        """Traverse the tree for a single time series instance."""
        if time_series is None:
            # Return None or a default prediction if input is None
            # Using the root node's prediction (or a global default) might be better
            # For now, returning None to indicate failure to predict.
            # Alternatively, could return tree.get(0, {}).get('prediction', 1) # Fallback
            return None

        node_id = 0  # Start at root

        # Traverse the tree until a leaf node is reached or traversal stops
        while node_id in tree:
            current_node = tree[node_id]

            # If it's a leaf node, return its prediction
            if current_node['is_leaf']:
                return current_node['prediction']

            # If it's an internal node, use the split info to decide which branch to follow
            split_info = current_node.get('split_on') # Use .get for safety
            children = current_node.get('children')

            # Ensure split info and children exist for internal nodes
            if split_info and children and len(children) > 0:
                measure_type, exemplars = split_info # split_info is (measure_type, {branch_id: exemplar_ts})

                # Handle case where exemplars might be empty (shouldn't happen in valid tree)
                if not exemplars:
                     print(f"WARNING: Node {node_id} has split_info but empty exemplars. Returning node prediction.")
                     return current_node.get('prediction', 1) # Fallback prediction

                # Calculate distance to ALL exemplars for this node's split
                min_dist_all_exemplars = float('inf')
                best_branch_id_all_exemplars = None

                for branch_id, exemplar_ts in exemplars.items():
                    # Calculate distance using the specified measure (placeholder: euclidean)
                    # In a real implementation, call a function that dispatches based on measure_type
                    d = euclidean_distance(time_series, exemplar_ts) # Use the distance function

                    if d < min_dist_all_exemplars:
                        min_dist_all_exemplars = d
                        best_branch_id_all_exemplars = branch_id

                # --- Enhanced Traversal Logic ---
                # Check if the child node corresponding to the nearest exemplar exists
                if best_branch_id_all_exemplars is not None and best_branch_id_all_exemplars in children:
                    # If the child exists, move to that child node
                    node_id = children[best_branch_id_all_exemplars]
                    # print(f"DEBUG: Node {current_node['node_id']}, nearest exemplar branch {best_branch_id_all_exemplars} exists, moving to child {node_id}") # Debug)
                else:
                    # If the child corresponding to the nearest exemplar does NOT exist (pruned branch),
                    # or if no nearest exemplar was found (e.g., all distances inf)
                    # find the nearest exemplar among the *existing* child branches and follow that path.
                    # print(f"DEBUG: Node {current_node['node_id']}, nearest exemplar branch {best_branch_id_all_exemplars} does not exist or no nearest found. Finding nearest among existing children.") # Debug)
                    min_dist_existing_children = float('inf')
                    next_node_id = None

                    # Iterate through the *existing* child branches
                    for existing_branch_id, existing_child_id in children.items():
                        # Find the exemplar time series for this existing branch from the original exemplars
                        # It's crucial that the branch_id used as the key in 'children' corresponds
                        # to the branch_id (exemplar label) in the 'exemplars' dictionary.
                        if existing_branch_id in exemplars:
                            existing_exemplar_ts = exemplars[existing_branch_id]
                            # Calculate distance to this existing branch's exemplar
                            d = euclidean_distance(time_series, existing_exemplar_ts) # Use the distance function

                            if d < min_dist_existing_children:
                                min_dist_existing_children = d
                                next_node_id = existing_child_id

                    # If a nearest existing child was found, move to that child node
                    if next_node_id is not None:
                        node_id = next_node_id
                        # print(f"DEBUG: Node {current_node['node_id']}, routed to nearest existing child {node_id} via branch {next_node_id}.") # Debug)
                    else:
                        # If no existing children were found or no nearest existing child determined,
                        # stop traversal and return the current node's prediction.
                        # This handles cases where time_series might be equidistant or invalid distances occur.
                        # print(f"DEBUG: Node {current_node['node_id']}, no nearest existing child found. Stopping traversal.") # Debug)
                        return current_node.get('prediction', 1) # Return prediction of current node, fallback to 1

            else:
                # If the node is internal but has no split info or children (shouldn't happen with correct training)
                # Or if it's marked internal but effectively a leaf due to pruning/lack of data
                # print(f"DEBUG: Node {current_node['node_id']} is internal-like but has no valid split info/children, returning prediction.") # Debug)
                # Return the prediction of the current node (should have been set if finalized as leaf)
                return current_node.get('prediction', 1) # Fallback prediction


        # If the loop finishes without returning (e.g., node_id not found, which indicates an issue)
        # Return a default fallback prediction
        # print(f"DEBUG: Traversal ended unexpectedly outside loop, final node_id {node_id}. Using default 1.") # Debug)
        # It might be better to return the prediction of the last valid node visited,
        # but for simplicity, using a default.
        last_node_pred = tree.get(node_id, {}).get('prediction', 1) # Get prediction of last node if possible
        return last_node_pred


    return traverse_tree


class GlobalProxTree:
    def __init__(self, spark, max_depth=5, min_samples=5, num_candidate_splits=5, num_exemplars_per_class=1):
        """
        Initialize the Global Proximity Tree

        Parameters:
        -----------
        spark : SparkSession
            The Spark session to use
            max_depth : int or None, default=5
            Maximum depth of the tree. If None, grows until min_samples or purity criteria met.
        min_samples : int
            Minimum number of samples required to split a node
        num_candidate_splits : int
            Number of random candidate splits to evaluate at each node.
        num_exemplars_per_class : int
            Number of exemplars to sample per class for each open node.
            (Used for sampling pool on driver, not per candidate split as in paper)
        """
        self.spark = spark
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.num_candidate_splits = num_candidate_splits
        # Note: num_exemplars_per_class here is used to sample a pool of exemplars
        # to the driver per node/label, not per candidate split.
        # The paper samples 1 exemplar per class *per candidate split*.
        self.num_exemplars_per_class = num_exemplars_per_class


        # Define the schema for data assigned to nodes
        self.assignment_schema = StructType([
            StructField("row_id", IntegerType(), False), # Add a unique row ID
            StructField("node_id", IntegerType(), False),
            StructField("time_series", ArrayType(DoubleType()), False),
            StructField("true_label", IntegerType(), False),
        ])

        # NOTE: TreeNode definition moved outside the class

        # Initialize the tree with a root node using the module-level TreeNode
        self.tree = {
            0: TreeNode( # Use module-level TreeNode definition
                node_id=0,
                parent_id=None,
                split_on=None, # Will store the chosen split info (measure, exemplars)
                is_leaf=False,
                prediction=None,
                children={}, # {branch_id: child_node_id}
            )
        }
        self._next_node_id = 1 # Counter for assigning new node IDs

        # Store the overall majority class for fallback prediction if needed
        self._overall_majority_class = None


    def _convert_to_time_series_format(self, df):
        """
        Convert wide dataframe (with each feature in its own column) to a dataframe
        with a single array column containing all features. Adds a unique row_id.
        Also handles the case where the input DataFrame might already have 'true_label'.

        Parameters:
        -----------
        df : Spark DataFrame
            Wide DataFrame with feature columns and 'label' or 'true_label' column,
            or DataFrame already containing 'time_series' column.

        Returns:
        --------
        Spark DataFrame
            DataFrame with 'row_id', 'time_series', and potentially 'true_label' columns.
        """
        print("DEBUG: _convert_to_time_series_format started.")

        # Add row_id if it doesn't exist
        if 'row_id' not in df.columns:
             print("DEBUG: Adding row_id.")
             df = df.withColumn("row_id", F.monotonically_increasing_id())
        else:
             print("DEBUG: row_id already exists.")

        # Check if 'time_series' column already exists
        if 'time_series' in df.columns:
            print("DEBUG: DataFrame already has 'time_series' column.")
            # Ensure 'true_label' exists if 'label' exists, otherwise keep as is
            if 'label' in df.columns and 'true_label' not in df.columns:
                 print("DEBUG: Renaming 'label' to 'true_label'.")
                 df = df.withColumnRenamed("label", "true_label").drop("label") # Rename and drop old
            elif 'true_label' in df.columns:
                 print("DEBUG: 'true_label' already exists.")
            else:
                 print("DEBUG: No 'label' or 'true_label' column found.")
            print("DEBUG: _convert_to_time_series_format finished (already formatted).")
            return df

        # --- Conversion Logic ---
        label_col_name = None
        if 'label' in df.columns:
            label_col_name = 'label'
        elif 'true_label' in df.columns:
            label_col_name = 'true_label'

        # Get all column names except label and row_id
        feature_cols = [col for col in df.columns if col not in [label_col_name, 'row_id']]

        print(f"DEBUG: Converting {len(feature_cols)} feature columns to 'time_series' array.")

        # Select columns for the new DataFrame
        select_expr = [
            F.col("row_id"),
            F.array(*[F.col(c) for c in feature_cols]).alias("time_series")
        ]

        # Add label column if it exists
        if label_col_name:
            select_expr.append(F.col(label_col_name).cast(IntegerType()).alias("true_label"))
            print(f"DEBUG: Including '{label_col_name}' as 'true_label'.")
        else:
            print("DEBUG: No label column found to include.")


        # Create the new DataFrame
        ts_df = df.select(*select_expr)


        # Show sample of converted data
        # print("DEBUG: Sample of converted DataFrame:") # Removed this line
        # ts_df.show(2, truncate=False) # Removed this line
        print("DEBUG: _convert_to_time_series_format finished (conversion done).")

        return ts_df


    def fit(self, df):
        """
        Fit the decision tree on the dataframe

        Parameters:
        -----------
        df : Spark DataFrame
            DataFrame with feature columns and 'label' or 'true_label' column

        Returns:
        --------
        self : GlobalProxTree
            The fitted tree
        """
        print("DEBUG: fit started.")
        # First, convert to time_series format if needed and add row_id
        # This now expects 'true_label' after conversion
        df = self._convert_to_time_series_format(df)

        # Ensure 'true_label' exists for fitting
        if 'true_label' not in df.columns:
             raise ValueError("Input DataFrame must have a 'label' or 'true_label' column for fitting.")

        # Calculate overall majority class for fallback prediction
        label_counts = df.groupBy("true_label").count().collect()
        if label_counts:
            # Handle potential ties in majority class (e.g., pick the smallest label)
            max_count = max(row['count'] for row in label_counts)
            majority_labels = [row['true_label'] for row in label_counts if row['count'] == max_count]
            self._overall_majority_class = min(majority_labels) # Deterministic tie-breaking
            print(f"DEBUG: Overall majority class calculated: {self._overall_majority_class}")
        else:
            self._overall_majority_class = None # Or set a default like 1?
            print("DEBUG: No data to calculate overall majority class.")


        # Initialize assignment dataframe with all rows at the root node
        # Select only necessary columns to minimize data size
        assign_df = (
            df
            .withColumn("node_id", F.lit(0).cast(IntegerType()))
            .select("row_id", "node_id", "time_series", "true_label")
            .cache()
        )
        initial_count = assign_df.count()
        if initial_count == 0:
             print("WARNING: Input DataFrame for fit is empty. Tree will not be built.")
             assign_df.unpersist()
             # Set root as leaf with default prediction
             self.tree[0] = self.tree[0]._replace(
                 is_leaf=True,
                 prediction=self._overall_majority_class if self._overall_majority_class is not None else 1
             )
             return self

        print(f"DEBUG: Initial assign_df created with {initial_count} rows at root node 0.")


        open_nodes = {0}
        current_depth = 0

        # Loop continues as long as there are open nodes AND (max_depth is None OR current_depth < max_depth)
        while open_nodes and (self.max_depth is None or current_depth < self.max_depth):
            print(f"\nDEBUG: === Starting tree level {current_depth} ===")
            # If no nodes to expand, stop
            if not open_nodes:
                print(f"DEBUG: No open_nodes at depth {current_depth}, stopping tree building.")
                break

            print(f"DEBUG: Open nodes at depth {current_depth}: {open_nodes}")

            # Filter assign_df to only include rows at the current open nodes
            current_level_df = assign_df.filter(F.col("node_id").isin(list(open_nodes))).cache()
            current_level_count = current_level_df.count()
            print(f"DEBUG: Filtered data for current level. Row count: {current_level_count}")

            # Check if any data exists for the current open nodes
            if current_level_count == 0:
                print(f"DEBUG: No data for open nodes at depth {current_depth}, stopping.")
                # Make these open_nodes leaves if they haven't been finalized
                for node_id in open_nodes:
                     if node_id in self.tree and not self.tree[node_id].is_leaf:
                          print(f"DEBUG: Making node {node_id} a leaf due to no data reaching it.")
                          # Prediction should ideally be based on parent or fallback
                          parent_id = self.tree[node_id].parent_id
                          parent_prediction = self.tree.get(parent_id, {}).get('prediction', self._overall_majority_class or 1)
                          self.tree[node_id] = self.tree[node_id]._replace(is_leaf=True, prediction=parent_prediction, children={})
                current_level_df.unpersist()
                break


            # --- Corrected Exemplar Sampling Logic (Driver-side) ---
            print("DEBUG: Sampling exemplars (driver-side).")
            sampled_exemplars = {} # {node_id: {true_label: [exemplar_ts1, exemplar_ts2, ...]}}

            # Get distinct (node_id, true_label) pairs present in the current level's data
            # Use .limit() to avoid collecting too much data if distinct pairs are numerous
            # A better approach for very large scale might involve approxQuantile or sampling
            node_label_pairs = current_level_df.select("node_id", "true_label").distinct().limit(10000).collect() # Limit distinct pairs collected
            if len(node_label_pairs) >= 10000:
                 print("WARNING: Reached limit of distinct (node_id, true_label) pairs collected for sampling. May affect exemplar diversity.")
            print(f"DEBUG: Found {len(node_label_pairs)} distinct (node_id, true_label) pairs for sampling.")

            for node_id, true_label in node_label_pairs:
                # Check if node is still considered open (might have been finalized)
                if node_id not in open_nodes:
                     continue

                print(f"DEBUG: Sampling exemplars for node {node_id}, label {true_label}.")
                # Filter the current level's data for this specific node and label
                node_label_df = current_level_df.filter((F.col("node_id") == node_id) & (F.col("true_label") == true_label))

                # Take a sample of rows for this node and label using RDD sampling for better distribution
                sample_fraction = min(1.0, self.num_exemplars_per_class / node_label_df.count()) if node_label_df.count() > 0 else 0.0
                # Use DataFrame sample - less precise count but avoids collect()
                sampled_rows_df = node_label_df.sample(withReplacement=False, fraction=sample_fraction, seed=42)
                # Limit collected data to avoid OOM
                sampled_rows = sampled_rows_df.select("time_series").limit(self.num_exemplars_per_class * 2).collect() # Collect slightly more due to fraction inaccuracy
                sampled_time_series = [row.time_series for row in sampled_rows][:self.num_exemplars_per_class] # Take the exact number needed


                if node_id not in sampled_exemplars:
                    sampled_exemplars[node_id] = {}
                sampled_exemplars[node_id][true_label] = sampled_time_series
                print(f"DEBUG: Sampled {len(sampled_time_series)} exemplars for node {node_id}, label {true_label}.")

            print(f"DEBUG: Finished sampling exemplars.")
            # --- End Corrected Exemplar Sampling ---


            # 2. Generate and evaluate candidate splits for each open node
            best_splits = {} # {node_id: (best_gini_gain, best_distance_measure, {branch_id: exemplar_ts})}
            nodes_to_make_leaves_this_iter = set() # Nodes that should become leaves in *this* iteration

            for node_id in list(open_nodes): # Iterate over a copy as we might modify open_nodes implicitly
                print(f"DEBUG: Evaluating splits for node {node_id}.")
                if node_id not in sampled_exemplars or not sampled_exemplars[node_id]:
                    print(f"DEBUG: No exemplars found for node {node_id}, making it a leaf.")
                    nodes_to_make_leaves_this_iter.add(node_id)
                    continue # Cannot split without exemplars

                node_data_df = current_level_df.filter(F.col("node_id") == node_id).cache()
                node_total_samples = node_data_df.count()

                # Check if node still has data after filtering
                if node_total_samples == 0:
                     print(f"DEBUG: Node {node_id} has 0 samples after filtering, making it a leaf.")
                     nodes_to_make_leaves_this_iter.add(node_id)
                     node_data_df.unpersist()
                     continue

                if node_total_samples < self.min_samples:
                    print(f"DEBUG: Node {node_id} has {node_total_samples} samples, below min_samples {self.min_samples}, making it a leaf.")
                    nodes_to_make_leaves_this_iter.add(node_id)
                    node_data_df.unpersist()
                    continue

                # Calculate parent Gini impurity (needs a collect)
                parent_label_counts_rows = node_data_df.groupBy("true_label").count().collect()
                # Convert Spark Rows to dictionary for Gini function
                parent_label_counts_dict = {row['true_label']: row['count'] for row in parent_label_counts_rows}
                parent_gini = self._calculate_gini_impurity(parent_label_counts_dict, node_total_samples)
                print(f"DEBUG: Node {node_id} parent Gini: {parent_gini}")

                # Check for pure node (Gini == 0)
                if parent_gini == 0.0:
                     print(f"DEBUG: Node {node_id} is pure (Gini=0), making it a leaf.")
                     nodes_to_make_leaves_this_iter.add(node_id)
                     node_data_df.unpersist()
                     continue


                best_gini_gain = -1.0
                best_split_info = None # (distance_measure, {branch_id: exemplar_ts})

                # Generate and evaluate candidate splits
                for i in range(self.num_candidate_splits):
                    print(f"DEBUG: Evaluating candidate split {i+1} for node {node_id}.")
                    # Sample a distance measure and parameters (simplified: using Euclidean)
                    distance_measure_type = "euclidean" # Placeholder

                    # Sample exemplars for this candidate split (one per class present in node_data_df)
                    # Get unique labels present in the sampled exemplars pool for this node
                    unique_labels_in_node_pool = list(sampled_exemplars.get(node_id, {}).keys())
                    if not unique_labels_in_node_pool:
                         print(f"WARNING: No labels in exemplar pool for node {node_id}. Cannot create split.")
                         continue # Skip this candidate split if no exemplars available


                    candidate_exemplars = {}
                    sampled_labels_count = 0
                    for label in unique_labels_in_node_pool:
                        if label in sampled_exemplars[node_id] and sampled_exemplars[node_id][label]:
                            # Pick one random exemplar for this label from the sampled pool for this node
                            candidate_exemplars[label] = random.choice(sampled_exemplars[node_id][label])
                            sampled_labels_count += 1
                        # else: # This case should ideally not happen if unique_labels_in_node_pool is derived correctly
                        #    print(f"WARNING: Label {label} in pool but no exemplars found for node {node_id}.")


                    if sampled_labels_count < 2:
                        print(f"DEBUG: Candidate split {i+1} for node {node_id} has {sampled_labels_count} unique exemplar labels, skipping (need >= 2).")
                        continue # Need at least two distinct branches/exemplars

                    print(f"DEBUG: Candidate split {i+1} exemplars (labels): {list(candidate_exemplars.keys())}")

                    # --- Modified Gini Calculation: Use RDD transformations ---
                    bc_candidate_exemplars = self.spark.sparkContext.broadcast(candidate_exemplars)

                    def map_to_branch_label_pair(row):
                        """Maps a row to its assigned branch ID and true label."""
                        exemplars = bc_candidate_exemplars.value
                        min_dist = float('inf')
                        assigned_branch_id = None # The label of the nearest exemplar

                        # Handle case where exemplars might be empty after broadcast (shouldn't happen)
                        if not exemplars: return (None, row.true_label)

                        for ex_lbl, ex_ts in exemplars.items():
                            d = euclidean_distance(row.time_series, ex_ts)
                            if d < min_dist:
                                min_dist = d
                                assigned_branch_id = ex_lbl

                        # If no nearest exemplar found (e.g., all inf distances), assign to None branch?
                        # Or assign randomly? For now, None.
                        return (assigned_branch_id, row.true_label)

                    # Apply the map to get (branch_id, true_label) pairs distributedly
                    branch_label_rdd = node_data_df.rdd.map(map_to_branch_label_pair)

                    # Group by branch_id and count labels within each branch distributedly
                    # Filter out None branch_id if it occurred
                    branch_label_counts_rdd = branch_label_rdd \
                        .filter(lambda pair: pair[0] is not None) \
                        .groupByKey() \
                        .mapValues(lambda labels: collections.Counter(labels))

                    # Collect the counts per branch to the driver for Gini calculation
                    # This is a potential bottleneck but necessary for weighted sum
                    branch_counts_collected = branch_label_counts_rdd.collect()
                    print(f"DEBUG: Branch label counts collected for candidate split {i+1}: {branch_counts_collected}")


                    # Calculate weighted Gini impurity on the driver
                    total_weighted_impurity = 0.0
                    total_samples_in_split = 0 # Recalculate based on collected counts

                    for branch_id, label_counts_dict in branch_counts_collected:
                         branch_total = sum(label_counts_dict.values())
                         total_samples_in_split += branch_total
                         branch_impurity = self._calculate_gini_impurity(label_counts_dict, branch_total)
                         total_weighted_impurity += (branch_total / node_total_samples) * branch_impurity # Weight by parent total

                    # Handle cases where total_samples_in_split might be 0 or less than node_total_samples
                    # if total_samples_in_split != node_total_samples:
                    #      print(f"WARNING: Sum of branch samples ({total_samples_in_split}) != node total ({node_total_samples}) for split {i+1}. Might be due to None branch assignments.")
                    #      # Adjust weighting? Or ignore this split? For now, proceed.


                    print(f"DEBUG: Total weighted impurity calculated for candidate split {i+1}: {total_weighted_impurity}")


                    gini_gain = parent_gini - total_weighted_impurity
                    print(f"DEBUG: Candidate split {i+1} Gini gain: {gini_gain}")

                    # Unpersist the broadcast variable
                    bc_candidate_exemplars.unpersist(blocking=False) # Non-blocking unpersist
                    # --- End Modified Gini Calculation ---


                    # Check if this is the best split so far
                    # Add a small tolerance for floating point comparisons
                    if gini_gain > best_gini_gain + 1e-9:
                        best_gini_gain = gini_gain
                        best_split_info = (distance_measure_type, candidate_exemplars)
                        print(f"DEBUG: Candidate split {i+1} is the best so far for node {node_id} with gain {best_gini_gain}.")


                node_data_df.unpersist() # Unpersist node data

                # Decide if the node should split
                # A split occurs if best_gini_gain is significantly positive
                if best_gini_gain > 1e-9: # Use tolerance for splitting decision
                    print(f"DEBUG: Node {node_id} has a positive Gini gain ({best_gini_gain}), attempting to split.")
                    best_splits[node_id] = (best_gini_gain, best_split_info[0], best_split_info[1])
                else:
                    print(f"DEBUG: Node {node_id} has non-positive or negligible Gini gain ({best_gini_gain}), making it a leaf.")
                    nodes_to_make_leaves_this_iter.add(node_id)


            # --- Finalize nodes marked as leaves *in this iteration* ---
            for node_id in nodes_to_make_leaves_this_iter:
                if node_id in self.tree and not self.tree[node_id].is_leaf:
                    print(f"DEBUG: Finalizing node {node_id} as a leaf (in iteration).")
                    # Need to calculate the prediction for this leaf node
                    # Collect label counts for this node from assign_df
                    # Use current_level_df if available and filtered for this node, otherwise query assign_df
                    leaf_data_df = current_level_df.filter(F.col("node_id") == node_id) if current_level_df else assign_df.filter(F.col("node_id") == node_id)
                    leaf_data_df = leaf_data_df.cache() # Cache for count and collect
                    leaf_count = leaf_data_df.count()

                    leaf_prediction = None
                    if leaf_count > 0:
                         leaf_label_counts_rows = leaf_data_df.groupBy("true_label").count().collect()
                         if leaf_label_counts_rows:
                              max_c = max(row['count'] for row in leaf_label_counts_rows)
                              majority_lbls = [row['true_label'] for row in leaf_label_counts_rows if row['count'] == max_c]
                              leaf_prediction = min(majority_lbls) # Deterministic tie-breaking
                         else: # Should not happen if count > 0
                              leaf_prediction = self._overall_majority_class or 1
                    else:
                         # If no data reached this node (e.g., due to filtering issues or empty parent branch)
                         # Fallback to parent's prediction or overall majority
                         parent_id = self.tree.get(node_id, {}).parent_id
                         if parent_id is not None and parent_id in self.tree:
                              parent_node = self.tree[parent_id]
                              # If parent became a leaf, use its prediction. Otherwise, might need fallback.
                              leaf_prediction = parent_node.prediction if parent_node.is_leaf else (self._overall_majority_class or 1)
                         else:
                              leaf_prediction = self._overall_majority_class or 1 # Fallback to overall or default
                         print(f"DEBUG: Node {node_id} had no data, using fallback prediction: {leaf_prediction}")

                    leaf_data_df.unpersist()

                    self.tree[node_id] = self.tree[node_id]._replace(is_leaf=True, prediction=leaf_prediction, children={}) # Clear children
                    print(f"DEBUG: Node {node_id} marked as leaf with prediction {leaf_prediction}.")
                    # Remove from open_nodes if it was there
                    open_nodes.discard(node_id)
            # --- End Finalization in Iteration ---


            # 3. Perform the best splits and update the tree structure (on driver)
            # and push rows down to the new child nodes (distributed)
            next_open = set() # Nodes that successfully split and will be processed in the next iteration
            if best_splits:
                print("DEBUG: Performing best splits and pushing rows down.")
                # Create a mapping from (parent_node_id, assigned_branch_id) to new_child_node_id
                split_mapping = {} # {(parent_id, assigned_branch_id): child_node_id}
                nodes_that_actually_split = set() # Track nodes that successfully create children

                # --- Pre-calculate branch assignments and counts for all splitting nodes ---
                # This avoids recalculating distances multiple times
                print("DEBUG: Pre-calculating branch assignments for all splitting nodes.")
                nodes_to_split_list = list(best_splits.keys())
                data_for_splitting_nodes_df = assign_df.filter(F.col("node_id").isin(nodes_to_split_list)).cache()

                # Broadcast the necessary split info (exemplars per node)
                bc_best_splits_exemplars = self.spark.sparkContext.broadcast(
                    {nid: split_info[2] for nid, split_info in best_splits.items()}
                )

                def map_row_to_assigned_branch(row):
                    """Assigns a row to a branch based on the best split for its node."""
                    node_id = row.node_id
                    time_series = row.time_series
                    all_exemplars_map = bc_best_splits_exemplars.value

                    if node_id not in all_exemplars_map:
                        # Should not happen if filtered correctly
                        return (row.row_id, node_id, None) # Keep original node_id, no branch assigned

                    exemplars = all_exemplars_map[node_id]
                    min_dist = float('inf')
                    assigned_branch_id = None

                    if not exemplars: return (row.row_id, node_id, None) # Handle empty exemplars

                    for ex_lbl, ex_ts in exemplars.items():
                        d = euclidean_distance(time_series, ex_ts)
                        if d < min_dist:
                            min_dist = d
                            assigned_branch_id = ex_lbl

                    return (row.row_id, node_id, assigned_branch_id)

                # Apply UDF/map to get assignments: RDD[(row_id, parent_node_id, assigned_branch_id)]
                assignments_rdd = data_for_splitting_nodes_df.rdd.map(map_row_to_assigned_branch)
                assignments_rdd.cache() # Cache assignments

                # Count samples per (parent_node_id, assigned_branch_id)
                # Result: {(parent_node_id, assigned_branch_id): count}
                branch_counts_map = assignments_rdd.map(lambda x: ((x[1], x[2]), 1)) \
                                                .reduceByKey(lambda a, b: a + b) \
                                                .collectAsMap()

                print(f"DEBUG: Pre-calculated branch counts: {branch_counts_map}")
                bc_best_splits_exemplars.unpersist(blocking=False)
                # --- End Pre-calculation ---


                for parent_id, (gain, measure, exemplars) in best_splits.items():
                    print(f"DEBUG: Processing best split for parent node {parent_id}.")
                    # Update the tree structure on the driver with the chosen split info
                    self.tree[parent_id] = self.tree[parent_id]._replace(split_on=(measure, exemplars))
                    print(f"DEBUG: Node {parent_id} split_on updated: measure={measure}, exemplars_labels={list(exemplars.keys())}.")

                    # Mark the parent node as INTERNAL tentatively
                    self.tree[parent_id] = self.tree[parent_id]._replace(is_leaf=False, prediction=None)
                    print(f"DEBUG: Node {parent_id} tentatively marked as INTERNAL.")

                    children_created_for_node = False
                    current_parent_children = {} # Store children created in this loop

                    # Use the pre-calculated counts
                    for (p_id, branch_id), count in branch_counts_map.items():
                         if p_id != parent_id: continue # Only process counts for the current parent
                         if branch_id is None: continue # Skip rows not assigned to a valid branch

                         # Only create a child node if the branch has enough samples
                         if count >= self.min_samples:
                              child_id = self._next_node_id
                              self._next_node_id += 1
                              print(f"DEBUG: Creating child node {child_id} for branch {branch_id} of parent {parent_id} (count={count}).")
                              # Use module-level TreeNode
                              self.tree[child_id] = TreeNode(
                                   node_id=child_id,
                                   parent_id=parent_id,
                                   split_on=None,
                                   is_leaf=False, # Initially internal
                                   prediction=None,
                                   children={},
                              )
                              # Add to the parent's children dict (temporary, update tree later)
                              current_parent_children[branch_id] = child_id
                              # Add to the global mapping used for pushing rows
                              split_mapping[(parent_id, branch_id)] = child_id
                              # Add the new child node to the set of nodes for the next iteration
                              next_open.add(child_id)
                              children_created_for_node = True
                              print(f"DEBUG: Added child {child_id} to parent {parent_id} children map for branch {branch_id}.")
                         else:
                              print(f"DEBUG: Branch {branch_id} for node {parent_id} has {count} samples, below min_samples. Not creating child node.")
                              # Rows assigned here will keep parent_id in the update step

                    # Update the parent node's children in the main tree structure
                    if children_created_for_node:
                         self.tree[parent_id] = self.tree[parent_id]._replace(children=current_parent_children)
                         nodes_that_actually_split.add(parent_id)
                         print(f"DEBUG: Updated children for node {parent_id}: {current_parent_children}")
                    else:
                         # If no children were created, finalize the parent as a leaf
                         print(f"DEBUG: Node {parent_id} had positive Gini gain but no branches met min_samples. Finalizing as a leaf.")
                         # Recalculate prediction based on data at this node
                         leaf_data_df = assign_df.filter(F.col("node_id") == parent_id).cache()
                         leaf_count = leaf_data_df.count()
                         leaf_prediction = None
                         if leaf_count > 0:
                              leaf_label_counts_rows = leaf_data_df.groupBy("true_label").count().collect()
                              if leaf_label_counts_rows:
                                   max_c = max(row['count'] for row in leaf_label_counts_rows)
                                   majority_lbls = [row['true_label'] for row in leaf_label_counts_rows if row['count'] == max_c]
                                   leaf_prediction = min(majority_lbls)
                              else:
                                   leaf_prediction = self._overall_majority_class or 1
                         else:
                              leaf_prediction = self._overall_majority_class or 1 # Fallback
                         leaf_data_df.unpersist()

                         self.tree[parent_id] = self.tree[parent_id]._replace(is_leaf=True, prediction=leaf_prediction, children={}, split_on=None) # Clear children & split_on
                         print(f"DEBUG: Node {parent_id} finalized as leaf with prediction {leaf_prediction}.")
                         # Ensure it's not in open_nodes for next iteration
                         open_nodes.discard(parent_id)


                # --- Update assign_df based on pre-calculated assignments ---
                if split_mapping: # Only proceed if some children were created
                    print("DEBUG: Updating assign_df with new node assignments.")
                    bc_split_mapping = self.spark.sparkContext.broadcast(split_mapping)

                    def map_assignment_to_new_node(assignment_tuple):
                        """Uses the assignment RDD and split_mapping to get the final (row_id, new_node_id)."""
                        row_id, parent_node_id, assigned_branch_id = assignment_tuple
                        mapping = bc_split_mapping.value
                        # Get the new child_id if a mapping exists, otherwise keep the parent_id
                        new_node_id = mapping.get((parent_node_id, assigned_branch_id), parent_node_id)
                        return (row_id, new_node_id)

                    # RDD[(row_id, new_node_id)]
                    new_assignments_rdd = assignments_rdd.map(map_assignment_to_new_node)

                    # Convert back to DataFrame: [Row(row_id=..., new_node_id=...)]
                    new_assignments_df = new_assignments_rdd.toDF(["row_id", "new_node_id"])

                    # Join the original assign_df with the new assignments
                    # Left join ensures we keep rows that didn't split
                    updated_assign_df = assign_df.join(
                        new_assignments_df,
                        on="row_id",
                        how="left"
                    )

                    # Update the node_id: use new_node_id if available, otherwise keep old node_id
                    final_assign_df = updated_assign_df.withColumn(
                        "node_id",
                        F.coalesce(F.col("new_node_id"), F.col("node_id")).cast(IntegerType())
                    ).select("row_id", "node_id", "time_series", "true_label") # Select final columns

                    # Cache the result and unpersist intermediates
                    final_assign_df.cache()
                    final_assign_df.count() # Action to trigger caching

                    assign_df.unpersist() # Unpersist old assign_df
                    data_for_splitting_nodes_df.unpersist()
                    assignments_rdd.unpersist()
                    bc_split_mapping.unpersist(blocking=False)

                    assign_df = final_assign_df # Update reference
                    print(f"DEBUG: assign_df updated for depth {current_depth+1}. New total rows: {assign_df.count()}")

                else:
                     print("DEBUG: No children created across all nodes in this iteration. assign_df remains unchanged.")
                     # Unpersist intermediate RDD if it was created
                     if 'assignments_rdd' in locals() and assignments_rdd.is_cached:
                          assignments_rdd.unpersist()
                     if 'data_for_splitting_nodes_df' in locals() and data_for_splitting_nodes_df.is_cached:
                          data_for_splitting_nodes_df.unpersist()


            else: # No best_splits found
                print("DEBUG: No nodes identified for splitting in this iteration.")


            # Unpersist data for the current level
            current_level_df.unpersist()

            # Update open_nodes for the next iteration
            open_nodes = next_open
            print(f"DEBUG: open_nodes for next level ({current_depth+1}): {open_nodes}")
            current_depth += 1 # Increment depth for the next iteration


        # --- Finalize any remaining internal nodes as leaves ---
        print("\nDEBUG: === Finalizing remaining internal nodes as leaves (end of fit) ===")
        all_node_ids = list(self.tree.keys()) # Get keys before potential modification

        nodes_finalized_end_of_fit = 0
        for node_id in all_node_ids:
            # Check if the node exists and is NOT already marked as a leaf
            if node_id in self.tree and not self.tree[node_id].is_leaf:
                nodes_finalized_end_of_fit += 1
                print(f"DEBUG: Finalizing node {node_id} as a leaf (end of fit).")

                # Need to calculate the prediction for this leaf node
                leaf_data_df = assign_df.filter(F.col("node_id") == node_id).cache()
                leaf_count = leaf_data_df.count()
                leaf_prediction = None

                if leaf_count > 0:
                     leaf_label_counts_rows = leaf_data_df.groupBy("true_label").count().collect()
                     if leaf_label_counts_rows:
                          max_c = max(row['count'] for row in leaf_label_counts_rows)
                          majority_lbls = [row['true_label'] for row in leaf_label_counts_rows if row['count'] == max_c]
                          leaf_prediction = min(majority_lbls)
                          print(f"DEBUG: Node {node_id} majority prediction: {leaf_prediction}")
                     else:
                          leaf_prediction = self._overall_majority_class or 1
                          print(f"DEBUG: Node {node_id} had data but no labels? Using overall majority: {leaf_prediction}")
                else:
                     # Fallback to parent's prediction or overall majority
                     parent_id = self.tree.get(node_id, {}).parent_id
                     if parent_id is not None and parent_id in self.tree:
                          parent_node = self.tree[parent_id]
                          leaf_prediction = parent_node.prediction if parent_node.is_leaf else (self._overall_majority_class or 1)
                     else:
                          leaf_prediction = self._overall_majority_class or 1 # Fallback to overall or default
                     print(f"DEBUG: Node {node_id} had no data, using fallback prediction: {leaf_prediction}")

                leaf_data_df.unpersist() # Unpersist after collecting counts

                # Update the node in the tree structure
                self.tree[node_id] = self.tree[node_id]._replace(is_leaf=True, prediction=leaf_prediction, children={}) # Clear children
                print(f"DEBUG: Node {node_id} marked as leaf with prediction {leaf_prediction}.")

        if nodes_finalized_end_of_fit > 0:
             print(f"DEBUG: Finalized {nodes_finalized_end_of_fit} nodes at the end of fit.")
        else:
             print("DEBUG: No remaining internal nodes needed finalization at the end of fit.")
        # --- End Finalization ---


        if assign_df.is_cached:
             assign_df.unpersist() # Unpersist the final assignment DataFrame
        print("DEBUG: fit finished.")
        return self

    def _calculate_gini_impurity(self, label_counts_dict, total_samples):
        """
        Calculates Gini impurity from a dictionary of label counts.

        Parameters:
        -----------
        label_counts_dict : dict {label: count}
            Counts of each label in the dataset or branch.
        total_samples : int
            Total number of samples in the node/branch.

        Returns:
        --------
        float : Gini impurity
        """
        if total_samples == 0:
            return 0.0 # Gini is 0 for empty set

        impurity = 1.0
        for label, count in label_counts_dict.items():
            if count > 0: # Avoid division by zero if total_samples is 0 (already handled)
                 probability_of_label = count / total_samples
                 impurity -= probability_of_label ** 2

        # Clamp impurity to [0, 1] to handle potential floating point inaccuracies
        return max(0.0, min(impurity, 1.0))


    def predict(self, df):
        """
        Make predictions using the trained tree. Input DataFrame can have features
        spread across columns or a single 'time_series' column. It may or may not
        have a 'label' or 'true_label' column.

        Parameters:
        -----------
        df : Spark DataFrame
            DataFrame with feature columns or 'time_series' column.

        Returns:
        --------
        Spark DataFrame : DataFrame with original identifier ('row_id'),
                          'time_series', and 'prediction' columns.
                          Includes 'true_label' if present in the input.
        """
        print("DEBUG: predict started.")
        # First, convert to time_series format if needed and ensure row_id exists
        df_prepared = self._convert_to_time_series_format(df)

        # Check if the tree has been fitted (at least root node exists and is finalized)
        if 0 not in self.tree or (self.tree[0].prediction is None and not self.tree[0].children):
             raise RuntimeError("Tree has not been fitted or is empty. Call fit() before predict() or load a model.")

        # --- Convert tree structure to a plain dictionary for broadcasting ---
        print("DEBUG: Converting tree structure to plain dictionary for broadcasting.")
        plain_tree_structure = {}
        for node_id, node in self.tree.items():
            # Ensure prediction is serializable (it should be int or None)
            prediction_val = node.prediction
            if prediction_val is not None:
                try:
                    prediction_val = int(prediction_val)
                except (ValueError, TypeError):
                    print(f"Warning: Prediction for node {node_id} is not an integer ({prediction_val}). Setting to None.")
                    prediction_val = None # Or handle as error

            plain_tree_structure[node_id] = {
                'node_id': node.node_id,
                'parent_id': node.parent_id,
                # Ensure split_on is also a plain structure (tuple of string and dict)
                # Time series within split_on should be lists/tuples of numbers
                'split_on': node.split_on,
                'is_leaf': node.is_leaf,
                'prediction': prediction_val,
                # Children dictionary keys (branch_id) and values (child_node_id) are plain types
                'children': node.children
            }

        # Broadcast the plain tree structure
        print("DEBUG: Broadcasting plain tree structure for prediction.")
        try:
             plain_tree_structure_broadcast = self.spark.sparkContext.broadcast(plain_tree_structure)
        except Exception as e:
             print(f"ERROR: Failed to broadcast tree structure: {e}")
             # Optionally serialize and print structure for debugging
             # try:
             #     import json
             #     print("Problematic structure (partial):", json.dumps(plain_tree_structure.get(0, {}), indent=2))
             # except Exception as json_e:
             #     print("Could not serialize structure to JSON for debugging:", json_e)
             raise

        # Create the prediction UDF using the broadcasted plain tree
        # Pass the broadcast variable to the function that defines the UDF
        # Handle potential None predictions from UDF by coalescing with a default
        prediction_udf = F.udf(predict_udf_func(plain_tree_structure_broadcast), IntegerType())

        # Apply the prediction UDF to each row
        predictions_df = df_prepared.withColumn(
             "prediction_raw",
             prediction_udf(F.col("time_series"))
        )

        # Coalesce None predictions with a default (e.g., overall majority or 1)
        default_pred = self._overall_majority_class if self._overall_majority_class is not None else 1
        predictions_df = predictions_df.withColumn(
             "prediction",
             F.coalesce(F.col("prediction_raw"), F.lit(default_pred)).cast(IntegerType())
        ).drop("prediction_raw")


        # Unpersist the broadcast variable after the prediction is done
        plain_tree_structure_broadcast.unpersist(blocking=False) # Non-blocking
        print("DEBUG: Plain tree structure broadcast unpersisted.")

        print("DEBUG: predict finished.")

        # Select the relevant output columns
        select_cols = ["row_id", "time_series"]
        if "true_label" in df_prepared.columns:
            select_cols.append("true_label")
        select_cols.append("prediction")

        return predictions_df.select(*select_cols)


    def print_tree(self):
        """
        Print a representation of the tree (driver-side).

        Returns:
        --------
        str : String representation of the tree
        """
        print("DEBUG: print_tree started.")
        if not self.tree:
             return "Tree is empty."

        output_lines = []
        def print_node_recursive(node_id, depth=0):
            if node_id not in self.tree:
                output_lines.append(f"{'  ' * depth}Node {node_id}: Does Not Exist")
                return

            node = self.tree[node_id]
            indent = "  " * depth # Use 2 spaces for indentation

            # Format split_on info nicely
            split_info_str = "None"
            if node.split_on:
                measure_type, exemplars = node.split_on
                # Print exemplar labels for brevity
                exemplar_labels = list(exemplars.keys()) if isinstance(exemplars, dict) else "InvalidFormat"
                split_info_str = f"measure={measure_type}, exemplars_labels={exemplar_labels}"


            # Node info line
            status = 'LEAF' if node.is_leaf else 'INTERNAL'
            pred_str = f"prediction={node.prediction}" if node.is_leaf else "prediction=N/A"
            line = f"{indent}Node {node_id} (Depth {depth}, Parent: {node.parent_id}): {status}, {pred_str}, split_on=[{split_info_str}]"
            output_lines.append(line)

            # Recursively print children if they exist
            if node.children and isinstance(node.children, dict):
                 output_lines.append(f"{indent}  Children:")
                 # Sort children by branch_id for consistent output
                 for branch_id, child_id in sorted(node.children.items()):
                      output_lines.append(f"{indent}    Branch {branch_id} -> Child {child_id}")
                      # Recurse only if the child node exists
                      if child_id in self.tree:
                           print_node_recursive(child_id, depth + 1) # Increase depth
                      else:
                           output_lines.append(f"{indent}      Node {child_id}: Does Not Exist (Error in tree structure)")
            elif not node.is_leaf and not node.children:
                 output_lines.append(f"{indent}  Children: None (Internal node with no children)")


        print_node_recursive(0)  # Start at root (node_id 0) at depth 0
        tree_str = "\n".join(output_lines)
        print("DEBUG: print_tree finished.")
        return tree_str

    def save_tree(self, directory_path):
        """
        Saves the trained tree structure and hyperparameters to a file using pickle.

        Parameters:
        -----------
        directory_path : str
            The directory where the model file should be saved.
            The filename will be generated based on hyperparameters.
        """
        print(f"DEBUG: save_tree started. Saving to directory: {directory_path}")

        # Ensure the tree has been fitted (basic check: root node exists and is finalized or has children)
        if 0 not in self.tree or (not self.tree[0].is_leaf and not self.tree[0].children):
             print("WARNING: Attempting to save an unfitted or potentially invalid tree (root is not leaf and has no children).")
             # Consider raising an error if saving an unfitted tree is strictly disallowed
             # raise RuntimeError("Cannot save an unfitted/invalid tree. Call fit() first.")

        # Create the directory if it doesn't exist
        try:
             os.makedirs(directory_path, exist_ok=True)
             print(f"DEBUG: Ensured directory exists: {directory_path}")
        except OSError as e:
             print(f"ERROR: Could not create directory {directory_path}: {e}")
             raise

        # Construct filename based on hyperparameters
        filename = (
            f"global_model_max_depth={self.max_depth}_"
            f"min_samples={self.min_samples}_"
            f"num_splits={self.num_candidate_splits}_"
            f"num_exemplars={self.num_exemplars_per_class}.pkl"
        )
        full_path = os.path.join(directory_path, filename)
        print(f"DEBUG: Saving model to file: {full_path}")

        # Prepare data to save
        # The tree dictionary contains TreeNode namedtuples, which are pickleable
        # when defined at the module level.
        save_data = {
            'tree': self.tree,
            'hyperparameters': {
                'max_depth': self.max_depth,
                'min_samples': self.min_samples,
                'num_candidate_splits': self.num_candidate_splits,
                'num_exemplars_per_class': self.num_exemplars_per_class,
            },
            'overall_majority_class': self._overall_majority_class,
            '_next_node_id': self._next_node_id # Save the next node ID counter
        }

        # Save using pickle
        try:
            with open(full_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"DEBUG: Tree saved successfully to {full_path}")
        except Exception as e:
            print(f"ERROR: Failed to save tree to {full_path} using pickle: {e}")
            traceback.print_exc() # Print stack trace for detailed error
            raise # Re-raise the exception


    @classmethod
    def load_tree(cls, spark, file_path):
        """
        Loads a trained tree structure and hyperparameters from a file saved by save_tree.

        Parameters:
        -----------
        spark : SparkSession
            The Spark session to associate with the loaded model.
        file_path : str
            The full path to the saved model file (.pkl).

        Returns:
        --------
        GlobalProxTree : An instance of the class with the loaded state.
        """
        print(f"DEBUG: load_tree started. Loading from: {file_path}")

        # Load data using pickle
        try:
            with open(file_path, 'rb') as f:
                # This requires the TreeNode namedtuple to be defined in the scope
                # where load_tree is called (which it is, at the module level).
                load_data = pickle.load(f)
            print("DEBUG: Model data loaded successfully from file.")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {file_path}")
            raise
        except pickle.UnpicklingError as e:
             print(f"ERROR: Failed to unpickle data from {file_path}. File might be corrupted or incompatible: {e}")
             raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading tree from {file_path}: {e}")
            traceback.print_exc()
            raise

        # Extract data
        loaded_tree = load_data.get('tree')
        loaded_hyperparams = load_data.get('hyperparameters')
        loaded_majority_class = load_data.get('overall_majority_class')
        loaded_next_node_id = load_data.get('_next_node_id', 1) # Default if not saved

        # Basic validation
        if not isinstance(loaded_tree, dict) or not isinstance(loaded_hyperparams, dict):
             raise ValueError("Loaded file is missing essential tree data or hyperparameters, or they are in the wrong format.")
        if 0 not in loaded_tree: # Check if root node exists
             raise ValueError("Loaded tree data does not contain the root node (node_id 0).")

        # Create a new instance of the class using loaded hyperparameters
        print("DEBUG: Creating new GlobalProxTree instance with loaded hyperparameters.")
        try:
             instance = cls(spark=spark, **loaded_hyperparams)
        except TypeError as e:
             print(f"ERROR: Mismatch between loaded hyperparameters and class constructor: {e}")
             print(f"Loaded hyperparameters: {loaded_hyperparams}")
             raise

        # Restore the state
        instance.tree = loaded_tree
        instance._overall_majority_class = loaded_majority_class
        instance._next_node_id = loaded_next_node_id # Restore node ID counter

        print(f"DEBUG: Tree state restored. Root node is_leaf: {instance.tree.get(0).is_leaf}, prediction: {instance.tree.get(0).prediction}")
        print("DEBUG: load_tree finished.")
        return instance
