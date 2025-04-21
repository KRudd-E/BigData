import random
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def show_compact(df, num_rows=5, num_cols=3):
    """
    Print a compact version of a Spark DataFrame:
      - Shows the first `num_rows` rows.
      - If there are many columns, displays only the first `num_cols` columns,
        then a placeholder, then the last `num_cols` columns.
    """
    # Collect the first few rows
    rows = df.take(num_rows)
    all_cols = df.columns

    if len(all_cols) > 2 * num_cols:
        # Select first few and last few columns
        display_cols = all_cols[:num_cols] + ["..."] + all_cols[-num_cols:]
    else:
        display_cols = all_cols

    # Print column names
    print(" | ".join(display_cols))
    
    # For each row, print the selected columns
    for row in rows:
        if len(all_cols) > 2 * num_cols:
            first_vals = [str(row[col]) for col in all_cols[:num_cols]]
            last_vals = [str(row[col]) for col in all_cols[-num_cols:]]
            print(" | ".join(first_vals) + " | ... | " + " | ".join(last_vals))
        else:
            print(" | ".join([str(row[col]) for col in all_cols]))


def randomSplit_dist(df: DataFrame, weights=[0.8, 0.2], seed=123) -> tuple:
    """
    Splits a Spark DataFrame into train/test sets based on partition-respecting random assignment.
    -> class imbalances
    """
    assert abs(sum(weights) - 1.0) < 1e-6 #Weights must sum to 1.0

    # Assign a random number per row
    df_with_rand = df.withColumn("_rand", F.rand(seed))

    # Split based on cutoff
    train_df = df_with_rand.filter(F.col("_rand") <= weights[0]).drop("_rand")
    test_df = df_with_rand.filter(F.col("_rand") > weights[0]).drop("_rand")

    return train_df, test_df

def randomSplit_stratified_via_sampleBy(df, label_col, weights=[0.8, 0.2], seed=123):
    
    """
    Splits a Spark DataFrame into train/test sets based on partition-Preserves per‑class proportions

    """
    
    assert abs(sum(weights) - 1.0) < 1e-6 # ensure that our weights must sum to 1.0
    train_frac = weights[0]

    # figure out all the distinct label values 
    labels = [row[label_col] for row in df.select(label_col) 
                                            .distinct()  # build a tiny DataFrame of unique labels
                                            .collect() # brings the list to the driver
                                            ]

    # build a dict: each label -> same fraction
    fractions = {dict_lbl: train_frac for dict_lbl in labels}

    # sample train set: Use Spark’s native stratified sampler
    train_df = df.stat.sampleBy(label_col, fractions, seed)
    # everything else is test
    test_df  = df.join(train_df, on=df.columns, how="left_anti")

    return train_df, test_df    