from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ===============================
# CONFIG
# ===============================
DEFAULT_METADATA_PATH = Path(r"D:\github\IOT_Project\data\metadata\processed_slices_metadata.csv")


# ===============================
# DATASET
# ===============================
class MRISliceDataset(Dataset):
    def __init__(self, metadata_csv, split="train", transform=None):
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.transform = transform

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)

        required_columns = ["slice_filepath", "label", "split", "status"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in metadata: {missing_columns}")

        # Keep only successful rows
        df = df[df["status"] == "success"].copy()

        # Filter by split
        df = df[df["split"] == split].copy()

        # Remove rows with missing filepaths
        df = df[df["slice_filepath"].notna()].copy()

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No data found for split='{split}' in {self.metadata_csv}")

        self.df = df

        print(f"✅ Loaded split='{self.split}' with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        slice_path = row["slice_filepath"]
        label = int(row["label"])

        image = np.load(slice_path).astype(np.float32)

        # Shape should be (H, W) -> convert to (1, H, W)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D slice, got shape {image.shape} at {slice_path}")

        image = np.expand_dims(image, axis=0)  # (1, H, W)

        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ===============================
# DATALOADER FACTORY
# ===============================
def create_dataloader(
    metadata_csv,
    split="train",
    batch_size=32,
    shuffle=None,
    num_workers=0,
    pin_memory=False,
    transform=None
):
    if shuffle is None:
        shuffle = True if split == "train" else False

    dataset = MRISliceDataset(
        metadata_csv=metadata_csv,
        split=split,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def create_all_dataloaders(
    metadata_csv=DEFAULT_METADATA_PATH,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    transform=None
):
    train_loader = create_dataloader(
        metadata_csv=metadata_csv,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=transform
    )

    val_loader = create_dataloader(
        metadata_csv=metadata_csv,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=transform
    )

    test_loader = create_dataloader(
        metadata_csv=metadata_csv,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=transform
    )

    return train_loader, val_loader, test_loader


# ===============================
# TEST MAIN
# ===============================
def main():
    print("🧪 TESTING MRI SLICE DATALOADER...\n")

    train_loader, val_loader, test_loader = create_all_dataloaders(
        metadata_csv=DEFAULT_METADATA_PATH,
        batch_size=16,
        num_workers=0,
        pin_memory=False
    )

    print("\n📊 Number of batches:")
    print(f"Train: {len(train_loader)}")
    print(f"Val:   {len(val_loader)}")
    print(f"Test:  {len(test_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))

    print("\n📦 Sample batch info:")
    print(f"Images shape: {images.shape}")   # expected: (B, 1, 128, 128)
    print(f"Labels shape: {labels.shape}")   # expected: (B,)
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")

    print("\n🎯 Dataloader test completed successfully.")


if __name__ == "__main__":
    main()