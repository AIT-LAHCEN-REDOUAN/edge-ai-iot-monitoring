from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
INPUT_METADATA_CSV = Path(r"D:\github\IOT_Project\data\metadata\processed_volumes_metadata.csv")
OUTPUT_METADATA_CSV = Path(r"D:\github\IOT_Project\data\metadata\processed_slices_metadata.csv")
OUTPUT_BASE_DIR = Path(r"D:\github\IOT_Project\data\processed\slices")

SLICE_START = 32
SLICE_END = 96   # exclusive
EXPECTED_SHAPE = (128, 128, 128)


# ===============================
# HELPERS
# ===============================
def make_slice_output_dir(split: str, label: int) -> Path:
    out_dir = OUTPUT_BASE_DIR / split / str(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ===============================
# MAIN
# ===============================
def main():
    print("🧩 EXTRACTING 2D SLICES FROM PREPROCESSED VOLUMES...\n")

    if not INPUT_METADATA_CSV.exists():
        raise FileNotFoundError(f"Input metadata file not found: {INPUT_METADATA_CSV}")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_METADATA_CSV)

    # Keep only successful volumes
    df = df[df["status"] == "success"].copy().reset_index(drop=True)

    print(f"📄 Successful processed volumes found: {len(df)}")

    records = []
    failed_volumes = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting slices"):
        subject_id = str(row["subject_id"])
        processed_filepath = str(row["processed_filepath"])
        folder_name = str(row["folder_name"])
        label = int(row["label"])
        split = str(row["split"])

        try:
            volume = np.load(processed_filepath)

            if volume.shape != EXPECTED_SHAPE:
                raise ValueError(f"Unexpected volume shape: {volume.shape}, expected: {EXPECTED_SHAPE}")

            output_dir = make_slice_output_dir(split, label)

            for slice_idx in range(SLICE_START, SLICE_END):
                slice_2d = volume[:, :, slice_idx].astype(np.float32, copy=False)

                slice_filename = f"{subject_id}_slice_{slice_idx:03d}.npy"
                slice_output_path = output_dir / slice_filename

                np.save(slice_output_path, slice_2d)

                records.append({
                    "subject_id": subject_id,
                    "volume_filepath": processed_filepath,
                    "slice_filepath": str(slice_output_path),
                    "folder_name": folder_name,
                    "label": label,
                    "split": split,
                    "slice_index": slice_idx,
                    "slice_shape_x": slice_2d.shape[0],
                    "slice_shape_y": slice_2d.shape[1],
                    "status": "success",
                    "error_message": None
                })

            del volume

        except Exception as e:
            failed_volumes.append((subject_id, processed_filepath, str(e)))

            records.append({
                "subject_id": subject_id,
                "volume_filepath": processed_filepath,
                "slice_filepath": None,
                "folder_name": folder_name,
                "label": label,
                "split": split,
                "slice_index": None,
                "slice_shape_x": None,
                "slice_shape_y": None,
                "status": "failed",
                "error_message": str(e)
            })

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUTPUT_METADATA_CSV, index=False)

    success_df = result_df[result_df["status"] == "success"].copy()
    failed_df = result_df[result_df["status"] == "failed"].copy()

    print("\n==============================")
    print("✅ SLICE EXTRACTION COMPLETED")
    print("==============================")
    print(f"Saved metadata to: {OUTPUT_METADATA_CSV}")
    print(f"Total slice records: {len(result_df)}")
    print(f"Successful slice records: {len(success_df)}")
    print(f"Failed volume records: {len(failed_df)}")

    print("\n📊 Slice split distribution:")
    print(success_df["split"].value_counts())

    print("\n📊 Slice label distribution:")
    print(success_df["label"].value_counts().sort_index())

    if len(failed_volumes) > 0:
        print("\n⚠️ First 10 failed volumes:")
        for subject_id, path, error_msg in failed_volumes[:10]:
            print(f"- {subject_id} | {path}")
            print(f"  Error: {error_msg}")

    print("\n🎯 Slice extraction finished.")


if __name__ == "__main__":
    main()