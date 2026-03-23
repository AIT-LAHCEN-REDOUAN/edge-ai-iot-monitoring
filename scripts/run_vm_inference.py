import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.baseline.resnet18_model import ResNet18BinaryClassifier

VM_NAME = os.getenv("VM_NAME", "vm1")

VM_PROFILES_PATH = Path("/app/results/deployment/vm_profiles.json")
SLICES_METADATA_PATH = Path("/app/data/metadata/processed_slices_metadata.csv")
BASELINE_MODEL_PATH = Path("/app/models/baseline/best_model.pth")

DEVICE = torch.device("cpu")


def convert_host_path_to_container_path(path_str: str) -> Path:
    """
    Convert Windows host paths stored in CSV metadata into Docker container paths.
    Example:
        D:\\github\\IOT_Project\\data\\processed\\slices\\...  ->  /app/data/processed/slices/...
    """
    if not path_str:
        raise ValueError("Empty path string received.")

    normalized = str(path_str)

    # Windows absolute path from host
    if normalized.startswith("D:\\github\\IOT_Project"):
        normalized = normalized.replace("D:\\github\\IOT_Project", "/app")

    # In case backslashes remain, convert them to Linux-style slashes
    normalized = normalized.replace("\\", "/")

    return Path(normalized)


def load_vm_profiles(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"VM profiles file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_slice(metadata_path: Path):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Slices metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    df = df[df["status"] == "success"].copy()
    df = df[df["split"] == "test"].copy()

    if df.empty:
        raise ValueError("No successful test slices found.")

    row = df.iloc[0]

    raw_slice_path = row["slice_filepath"]
    slice_path = convert_host_path_to_container_path(raw_slice_path)
    label = int(row["label"])

    if not slice_path.exists():
        raise FileNotFoundError(
            f"Slice file not found inside container.\n"
            f"Original metadata path: {raw_slice_path}\n"
            f"Resolved container path: {slice_path}"
        )

    image = np.load(slice_path).astype(np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {image.shape}")

    image = np.expand_dims(image, axis=0)   # (1, H, W)
    image = np.expand_dims(image, axis=0)   # (1, 1, H, W)

    tensor = torch.from_numpy(image).float().to(DEVICE)
    return tensor, label, str(slice_path)


def build_model():
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    print(f"=== RUNNING CONTAINER INFERENCE FOR {VM_NAME.upper()} ===")

    vm_profiles = load_vm_profiles(VM_PROFILES_PATH)

    if VM_NAME not in vm_profiles:
        raise ValueError(f"{VM_NAME} not found in vm_profiles.json")

    vm_info = vm_profiles[VM_NAME]

    print(f"Profile name: {vm_info['profile_name']}")
    print(f"Selected technique: {vm_info['selected_technique']}")
    print(f"CPU limit: {vm_info['cpu_limit']}")
    print(f"Memory limit (MB): {vm_info['memory_limit_mb']}")
    print(f"Planned model size (MB): {vm_info['effective_size_mb']:.4f}")
    print(f"Planned accuracy: {vm_info['accuracy']:.6f}")

    x, true_label, slice_path = load_sample_slice(SLICES_METADATA_PATH)
    model = build_model()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item())

    print(f"Sample slice: {slice_path}")
    print(f"True label: {true_label}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.6f}")
    print("=== CONTAINER INFERENCE FINISHED ===")


if __name__ == "__main__":
    main()