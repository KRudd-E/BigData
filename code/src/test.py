# demo.py – unit-test GlobalModelManager on dummy data
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import os
import global_model_manager # Make sure this module is accessible

# ────────────────────────────────────────────────────────────────
# make the module visible to the Spark workers
# ────────────────────────────────────────────────────────────────



spark = ( SparkSession.builder
                .appName("LocalPipeline-pushed")
                # ────────────────────────────────────────────────────────────────
                #  CPU: use all physical cores + a few HT threads, but not 100 %
                # ────────────────────────────────────────────────────────────────
                .master("local[14]")                 # 8 phys. + 6 HT ≈ 87 % CPU
                .config("spark.task.cpus", "1")      # keep 1-thread tasks
                # ────────────────────────────────────────────────────────────────
                #  Heap: 26 GB for Spark + 4 GB off-heap = 30 GB total JVM budget
                # ────────────────────────────────────────────────────────────────
                .config("spark.driver.memory", "26g")
                .config("spark.memory.offHeap.enabled", "true")
                .config("spark.memory.offHeap.size",  "4g")
                .config("spark.driver.maxResultSize", "8g")
                # Tune the unified-memory manager
                .config("spark.memory.fraction", "0.75")        # storage+execution
                .config("spark.memory.storageFraction", "0.30") # cache vs. execution
                # ────────────────────────────────────────────────────────────────
                #  Shuffle + adaptive execution
                # ────────────────────────────────────────────────────────────────
                .config("spark.sql.shuffle.partitions", "32")   # ≈ 2× active cores
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                # ────────────────────────────────────────────────────────────────
                #  Arrow + GC
                # ────────────────────────────────────────────────────────────────
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config(
                    "spark.driver.extraJavaOptions",
                    # G1GC with a higher heap occupancy trigger & larger regions
                    "-XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:InitiatingHeapOccupancyPercent=40"
                )
                .getOrCreate()
            )
            
# --- UPDATED: Change log level for debugging ---
spark.sparkContext.setLogLevel("ERROR") # Changed from "ERROR"
# ------------------------------------------------

# Ensure the global_model_manager.py file is sent to worker nodes
spark.sparkContext.addPyFile(os.path.abspath(global_model_manager.__file__))

# ----------------------------------------------------------------
# dummy data
# ----------------------------------------------------------------
train_rows = [
        # Class 1 (around 0,0)
        (0.1, 0.1, 1), (0.2, 0.3, 1), (0.5, 0.2, 1), (0.3, 0.4, 1), (0.0, 0.0, 1),
        (0.4, 0.1, 1), (0.1, 0.5, 1), (0.6, 0.3, 1), (0.2, 0.2, 1), (0.3, 0.1, 1),
        (0.0, 0.4, 1), (0.5, 0.5, 1), (0.1, 0.2, 1), (0.4, 0.4, 1), (0.2, 0.0, 1),

        # Class 2 (around 5,5)
        (5.1, 5.1, 2), (5.3, 5.2, 2), (5.0, 5.5, 2), (5.4, 5.3, 2), (5.5, 5.0, 2),
        (5.2, 5.4, 2), (5.0, 5.1, 2), (5.3, 5.0, 2), (5.4, 5.4, 2), (5.1, 5.3, 2),
        (5.2, 5.0, 2), (5.0, 5.2, 2), (5.3, 5.3, 2), (5.4, 5.1, 2), (5.1, 5.4, 2),

        # Class 3 (around 10,10)
        (10.1, 10.1, 3), (10.2, 10.3, 3), (10.5, 10.2, 3), (10.3, 10.4, 3), (10.0, 10.0, 3),
        (10.4, 10.1, 3), (10.1, 10.5, 3), (10.6, 10.3, 3), (10.2, 10.2, 3), (10.3, 10.1, 3),
        (10.0, 10.4, 3), (10.5, 10.5, 3), (10.1, 10.2, 3), (10.4, 10.4, 3), (10.2, 10.0, 3),
    ]

test_rows = [
        # Test points clearly within clusters
        (0.2, 0.3, 1), (0.4, 0.1, 1), # Class 1
        (5.3, 5.4, 2), (5.1, 5.2, 2), # Class 2
        (10.2, 10.3, 3), (10.4, 10.1, 3), # Class 3

        # Test points between clusters (should be classified to nearest)
        (2.5, 2.5, 1), # Between 1 and 2
        (7.5, 7.5, 2), # Between 2 and 3
        (5.0, 0.0, 1), # Far from all, likely closest to 1 or 2 depending on exemplars
        (0.0, 5.0, 1), # Far from all, likely closest to 1 or 2 depending on exemplars
        (5.0, 10.0, 3), # Between 2 and 3
        (10.0, 5.0, 2), # Between 2 and 3
    ]
cols = ["f0", "f1", "label"]
train_df = spark.createDataFrame([Row(*r) for r in train_rows], cols)
test_df = spark.createDataFrame([Row(*r) for r in test_rows], cols)

# ----------------------------------------------------------------
# config (only the part needed by the tree)
# ----------------------------------------------------------------
gconf = {
    "tree_params": {
        "n_splitters": 5,
        "max_depth": 1,
        "min_samples_split": 2, # tiny data ⇒ let it split
        "random_state": 123,
    }
}

# ----------------------------------------------------------------
# train / predict
# ----------------------------------------------------------------
# Ensure GlobalModelManager is imported from your separate file
from global_model_manager import GlobalModelManager

gpt = GlobalModelManager(spark, gconf).fit(train_df)

print("\n=== TREE ===")
print(gpt.print_tree())

pred = gpt.predict(test_df).cache()
pred.select("row_id", "true_label", "prediction").show()

acc = pred.select(
    (F.col("true_label") == F.col("prediction")).cast("int").alias("ok")
).agg(F.avg("ok")).first()["avg(ok)"]
print(f"Accuracy = {acc:.3f}")

spark.stop()