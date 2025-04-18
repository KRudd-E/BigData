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
    """
    assert abs(sum(weights) - 1.0) < 1e-6 #Weights must sum to 1.0

    # Assign a random number per row
    df_with_rand = df.withColumn("_rand", F.rand(seed))

    # Split based on cutoff
    train_df = df_with_rand.filter(F.col("_rand") <= weights[0]).drop("_rand")
    test_df = df_with_rand.filter(F.col("_rand") > weights[0]).drop("_rand")

    return train_df, test_df
    