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