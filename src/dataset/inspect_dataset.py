from pathlib import Path
from collections import Counter
import nibabel as nib

# ===============================
# CONFIG
# ===============================
DATASET_PATH = Path(r"D:\github\IOT_Project\dataset")

FOLDERS = [
    "mr_IXI_480i",
    "mri_IXI_480_2"
]

VALID_EXTENSIONS = [".nii", ".gz"]


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


def inspect_folder(folder_path: Path, max_files_to_inspect: int = None):
    print(f"\n📂 Inspecting: {folder_path}")

    nii_files = get_nii_files(folder_path)
    print(f"➡️ Number of NIfTI files: {len(nii_files)}")

    shape_counter = Counter()
    dtype_counter = Counter()
    errors = []

    files_to_check = nii_files if max_files_to_inspect is None else nii_files[:max_files_to_inspect]

    for file in files_to_check:
        try:
            img = nib.load(str(file))

            # IMPORTANT:
            # do NOT call get_fdata() here to avoid loading full volume into RAM
            shape = img.shape
            dtype = img.get_data_dtype()

            shape_counter[shape] += 1
            dtype_counter[str(dtype)] += 1

        except Exception as e:
            errors.append((file.name, str(e)))

    return {
        "total_files": len(nii_files),
        "checked_files": len(files_to_check),
        "shape_counter": shape_counter,
        "dtype_counter": dtype_counter,
        "errors": errors
    }


def print_counter(title, counter_obj):
    print(f"\n📊 {title}")
    if not counter_obj:
        print("No data found.")
        return

    for item, count in counter_obj.most_common():
        print(f"{item} → {count}")


# ===============================
# MAIN
# ===============================
def main():
    print("🔍 DATASET INSPECTION STARTED\n")

    global_total = 0
    global_checked = 0
    all_errors = []

    for folder in FOLDERS:
        folder_path = DATASET_PATH / folder

        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            continue

        stats = inspect_folder(folder_path, max_files_to_inspect=None)

        global_total += stats["total_files"]
        global_checked += stats["checked_files"]
        all_errors.extend(stats["errors"])

        print_counter("Shape Summary", stats["shape_counter"])
        print_counter("Dtype Summary", stats["dtype_counter"])

        if stats["errors"]:
            print(f"\n⚠️ Errors in {folder}: {len(stats['errors'])}")
            for filename, error_msg in stats["errors"][:10]:
                print(f" - {filename}: {error_msg}")
        else:
            print(f"\n✅ No reading errors in {folder}")

    print("\n==============================")
    print("📌 GLOBAL SUMMARY")
    print("==============================")
    print(f"Total NIfTI files found: {global_total}")
    print(f"Total files checked: {global_checked}")
    print(f"Total errors: {len(all_errors)}")

    if all_errors:
        print("\n⚠️ First 10 global errors:")
        for filename, error_msg in all_errors[:10]:
            print(f" - {filename}: {error_msg}")

    print("\n🎯 Inspection completed.")


if __name__ == "__main__":
    main()