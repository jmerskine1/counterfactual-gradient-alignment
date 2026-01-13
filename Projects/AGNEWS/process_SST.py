import pandas as pd
import os

def process_counterfactual_csv(
    input_csv, output_dir="data/paired", test_size=0.15, dev_size=0.15
):
    """
    Convert a counterfactual dataset into sentiment-labeled rows,
    keeping original & counterfactual contiguous,
    and split sequentially (no shuffle) into train/dev/test TSVs.
    """

    # Load CSV
    df = pd.read_csv(input_csv)

    n = len(df)
    n_test = int(n * test_size)
    n_dev = int(n * dev_size)
    n_train = n - n_test - n_dev

    splits = {
        "train": df.iloc[:n_train],
        "dev": df.iloc[n_train:n_train+n_dev],
        "test": df.iloc[n_train+n_dev:],
    }

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_df in splits.items():
        records = []
        for idx, row in split_df.iterrows():
            # Original
            records.append({
                "Sentiment": "Negative" if row["original_score"] < 0.5 else "Positive",
                "Text": row["original_text"],
                "batch_id": idx
            })
            # Counterfactual
            records.append({
                "Sentiment": "Negative" if row["counterfactual_score"] < 0.5 else "Positive",
                "Text": row["counterfactual_text"],
                "batch_id": idx
            })

        flat_df = pd.DataFrame(records)

        # Save TSV with contiguous rows
        out_path = os.path.join(output_dir, f"{split_name}_paired.tsv")
        flat_df.to_csv(out_path, sep="\t", index=False)

    print(f"✅ Saved splits to {output_dir}/ (train/dev/test)")