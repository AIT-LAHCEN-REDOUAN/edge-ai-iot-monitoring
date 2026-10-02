from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG
# ===============================
METADATA_PATH = Path(r"D:\github\IOT_Project\data\metadata\dataset_inventory.csv")
OUTPUT_DIR = Path(r"D:\github\IOT_Project\data\metadata")

TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
VAL_CSV = OUTPUT_DIR / "val_split.csv"
TEST_CSV = OUTPUT_DIR / "test_split.csv"
FULL_SPLIT_CSV = OUTPUT_DIR / "dataset_with_splits.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ===============================
# MAIN
# ===============================
def main():
    print("✂️ SPLITTING DATASET...\n")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)

    print(f"📄 Loaded metadata: {len(df)} rows")

    required_columns = ["subject_id", "filepath", "label", "folder_name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata: {missing_columns}")

    # First split: train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        stratify=df["label"],
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Second split: temp -> val (15%) + test (15%)
    # Since temp is 30%, splitting it in half gives 15% / 15%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Save separate files
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    # Save combined file
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=["split", "label", "subject_id"]).reset_index(drop=True)
    full_df.to_csv(FULL_SPLIT_CSV, index=False)

    # ===============================
    # REPORTING
    # ===============================
    print("==============================")
    print("✅ SPLIT COMPLETED")
    print("==============================")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}")
    print(f"Full CSV:  {FULL_SPLIT_CSV}")

    print("\n📊 Split sizes:")
    print(f"Train: {len(train_df)} ({len(train_df) / len(df):.2%})")
    print(f"Val:   {len(val_df)} ({len(val_df) / len(df):.2%})")
    print(f"Test:  {len(test_df)} ({len(test_df) / len(df):.2%})")

    print("\n📊 Class distribution per split:")
    print("\nTrain:")
    print(train_df["label"].value_counts().sort_index())

    print("\nVal:")
    print(val_df["label"].value_counts().sort_index())

    print("\nTest:")
    print(test_df["label"].value_counts().sort_index())

    print("\n📊 Folder distribution per split:")
    print("\nTrain:")
    print(train_df["folder_name"].value_counts())

    print("\nVal:")
    print(val_df["folder_name"].value_counts())

    print("\nTest:")
    print(test_df["folder_name"].value_counts())

    print("\n🎯 Dataset splitting completed successfully.")


if __name__ == "__main__":
    main()