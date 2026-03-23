from pathlib import Path
import pandas as pd
import nibabel as nib

# ===============================
# CONFIG
# ===============================
DATASET_PATH = Path(r"D:\github\IOT_Project\dataset")
OUTPUT_DIR = Path(r"D:\github\IOT_Project\data\metadata")
OUTPUT_CSV = OUTPUT_DIR / "dataset_inventory.csv"

CLASS_MAPPING = {
    "mr_IXI_480i": 0,
    "mri_IXI_480_2": 1
}


# ===============================
# HELPERS
# ===============================
def get_nii_files(folder_path: Path):
    files = []
    for file in folder_path.iterdir():
        if file.is_file():
            if file.suffix == ".nii" or file.name.endswith(".nii.gz"):
                files.append(file)
    return sorted(files)


def extract_subject_id(filename: str) -> str:
    """
    Extract a subject identifier from filename.
    For now, we use the filename without extension as unique subject_id.
    Later, if needed, we can refine this rule.
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    return filename


def collect_file_info(folder_name: str, file_path: Path):
    try:
        img = nib.load(str(file_path))

        return {
            "subject_id": extract_subject_id(file_path.name),
            "filename": file_path.name,
            "filepath": str(file_path),
            "folder_name": folder_name,
            "label": CLASS_MAPPING[folder_name],
            "shape_x": img.shape[0] if len(img.shape) > 0 else None,
            "shape_y": img.shape[1] if len(img.shape) > 1 else None,
            "shape_z": img.shape[2] if len(img.shape) > 2 else None,
            "ndim": len(img.shape),
            "dtype": str(img.get_data_dtype())
        }
    except Exception as e:
        return {
            "subject_id": extract_subject_id(file_path.name),
            "filename": file_path.name,
            "filepath": str(file_path),
            "folder_name": folder_name,
            "label": CLASS_MAPPING[folder_name],
            "shape_x": None,
            "shape_y": None,
            "shape_z": None,
            "ndim": None,
            "dtype": None,
            "error": str(e)
        }


# ===============================
# MAIN
# ===============================
def main():
    print("📄 BUILDING DATASET METADATA...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for folder_name, label in CLASS_MAPPING.items():
        folder_path = DATASET_PATH / folder_name

        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            continue

        nii_files = get_nii_files(folder_path)
        print(f"📂 {folder_name} -> {len(nii_files)} files found")

        for file_path in nii_files:
            record = collect_file_info(folder_name, file_path)
            records.append(record)

    df = pd.DataFrame(records)

    # ensure consistent column order
    preferred_columns = [
        "subject_id",
        "filename",
        "filepath",
        "folder_name",
        "label",
        "shape_x",
        "shape_y",
        "shape_z",
        "ndim",
        "dtype",
        "error"
    ]

    for col in preferred_columns:
        if col not in df.columns:
            df[col] = None

    df = df[preferred_columns]

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n==============================")
    print("✅ METADATA CREATED")
    print("==============================")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")

    print("\n📊 Class distribution:")
    print(df["folder_name"].value_counts())

    print("\n📊 Label distribution:")
    print(df["label"].value_counts())

    print("\n📊 Shape distribution (top 10):")
    print(
        df[["shape_x", "shape_y", "shape_z"]]
        .value_counts()
        .head(10)
    )

    error_count = df["error"].notna().sum()
    print(f"\n⚠️ Files with errors: {error_count}")

    print("\n🎯 Metadata build completed.")


if __name__ == "__main__":
    main()