from pathlib import Path
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
INPUT_CSV = Path(r"D:\github\IOT_Project\data\metadata\dataset_with_splits.csv")
OUTPUT_METADATA_CSV = Path(r"D:\github\IOT_Project\data\metadata\processed_volumes_metadata.csv")
OUTPUT_BASE_DIR = Path(r"D:\github\IOT_Project\data\processed\volumes")

TARGET_SHAPE = (128, 128, 128)
EPSILON = 1e-8


# ===============================
# HELPERS
# ===============================
def resize_2d_slice(slice_2d: np.ndarray, target_hw=(128, 128)) -> np.ndarray:
    h, w = slice_2d.shape
    target_h, target_w = target_hw

    zoom_factors = (target_h / h, target_w / w)
    resized = zoom(slice_2d, zoom=zoom_factors, order=1)
    return resized.astype(np.float32, copy=False)


def resize_along_z(volume_xy_resized: np.ndarray, target_z: int) -> np.ndarray:
    current_z = volume_xy_resized.shape[2]
    zoom_factors = (1.0, 1.0, target_z / current_z)
    resized = zoom(volume_xy_resized, zoom=zoom_factors, order=1)
    return resized.astype(np.float32, copy=False)


def preprocess_volume_streaming(file_path: str, target_shape=(128, 128, 128)) -> tuple[np.ndarray, tuple]:
    """
    Memory-safe preprocessing:
    1) open NIfTI with nibabel proxy
    2) compute global min/max slice by slice
    3) normalize + resize each 2D slice
    4) resize along z-axis
    """
    img = nib.load(file_path, mmap=True, keep_file_open=False)
    proxy = img.dataobj
    original_shape = img.shape

    if len(original_shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {original_shape}")

    x, y, z = original_shape
    target_x, target_y, target_z = target_shape

    # -------------------------------
    # Pass 1: compute global min/max
    # -------------------------------
    global_min = np.inf
    global_max = -np.inf

    for k in range(z):
        slice_2d = np.asarray(proxy[:, :, k], dtype=np.float32)
        s_min = float(slice_2d.min())
        s_max = float(slice_2d.max())

        if s_min < global_min:
            global_min = s_min
        if s_max > global_max:
            global_max = s_max

        del slice_2d

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Invalid volume min/max values")

    # If constant image
    if (global_max - global_min) < EPSILON:
        resized_xy = np.zeros((target_x, target_y, z), dtype=np.float32)
    else:
        # ---------------------------------------------
        # Pass 2: normalize + resize each slice in XY
        # ---------------------------------------------
        resized_xy = np.empty((target_x, target_y, z), dtype=np.float32)

        for k in range(z):
            slice_2d = np.asarray(proxy[:, :, k], dtype=np.float32)
            slice_2d = (slice_2d - global_min) / (global_max - global_min)
            slice_2d = np.clip(slice_2d, 0.0, 1.0)

            resized_slice = resize_2d_slice(slice_2d, target_hw=(target_x, target_y))
            resized_xy[:, :, k] = resized_slice

            del slice_2d
            del resized_slice

    # -------------------------------
    # Resize along Z
    # -------------------------------
    final_volume = resize_along_z(resized_xy, target_z=target_z)

    del resized_xy
    gc.collect()

    return final_volume, original_shape


def make_output_path(split: str, subject_id: str) -> Path:
    split_dir = OUTPUT_BASE_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{subject_id}.npy"


# ===============================
# MAIN
# ===============================
def main():
    print("🧠 PREPROCESSING NIfTI VOLUMES (STREAMING MODE)...\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_columns = ["subject_id", "filepath", "label", "folder_name", "split"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing volumes"):
        subject_id = str(row["subject_id"])
        filepath = str(row["filepath"])
        label = int(row["label"])
        folder_name = str(row["folder_name"])
        split = str(row["split"])

        try:
            volume, original_shape = preprocess_volume_streaming(
                filepath,
                target_shape=TARGET_SHAPE
            )

            output_path = make_output_path(split, subject_id)
            np.save(output_path, volume)

            records.append({
                "subject_id": subject_id,
                "original_filepath": filepath,
                "processed_filepath": str(output_path),
                "folder_name": folder_name,
                "label": label,
                "split": split,
                "original_shape_x": original_shape[0],
                "original_shape_y": original_shape[1],
                "original_shape_z": original_shape[2],
                "processed_shape_x": TARGET_SHAPE[0],
                "processed_shape_y": TARGET_SHAPE[1],
                "processed_shape_z": TARGET_SHAPE[2],
                "status": "success",
                "error_message": None
            })

        except Exception as e:
            records.append({
                "subject_id": subject_id,
                "original_filepath": filepath,
                "processed_filepath": None,
                "folder_name": folder_name,
                "label": label,
                "split": split,
                "original_shape_x": None,
                "original_shape_y": None,
                "original_shape_z": None,
                "processed_shape_x": None,
                "processed_shape_y": None,
                "processed_shape_z": None,
                "status": "failed",
                "error_message": str(e)
            })

        finally:
            if "volume" in locals():
                del volume
            gc.collect()

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUTPUT_METADATA_CSV, index=False)

    success_df = result_df[result_df["status"] == "success"].copy()
    failed_df = result_df[result_df["status"] == "failed"].copy()

    print("\n==============================")
    print("✅ PREPROCESSING COMPLETED")
    print("==============================")
    print(f"Saved metadata to: {OUTPUT_METADATA_CSV}")
    print(f"Total rows: {len(result_df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Failed: {len(failed_df)}")

    if len(failed_df) > 0:
        print("\n⚠️ First 10 errors:")
        for _, row in failed_df.head(10).iterrows():
            print(f"- {row['subject_id']} | {row['original_filepath']}")
            print(f"  Error: {row['error_message']}")

    print("\n📊 Successful split distribution:")
    print(success_df["split"].value_counts())

    print("\n📊 Failed split distribution:")
    print(failed_df["split"].value_counts())

    print("\n🎯 Volume preprocessing finished.")


if __name__ == "__main__":
    main()