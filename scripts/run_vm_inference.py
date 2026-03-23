import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.baseline.resnet18_model import ResNet18BinaryClassifier

VM_NAME = "vm1"

VM_PROFILES_PATH = Path("/app/results/deployment/vm_profiles.json")
SLICES_METADATA_PATH = Path("/app/data/metadata/processed_slices_metadata.csv")
BASELINE_MODEL_PATH = Path("/app/models/baseline/best_model.pth")

DEVICE = torch.device("cpu")


def load_vm_profiles(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_slice(metadata_path: Path):
    df = pd.read_csv(metadata_path)
    df = df[df["status"] == "success"].copy()
    df = df[df["split"] == "test"].copy()

    row = df.iloc[0]
    slice_path = Path(row["slice_filepath"])
    label = int(row["label"])

    image = np.load(slice_path).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    tensor = torch.from_numpy(image).float().to(DEVICE)
    return tensor, label, str(slice_path)


def main():
    print(f"Running inference inside container for {VM_NAME}")

    vm_profiles = load_vm_profiles(VM_PROFILES_PATH)
    vm_info = vm_profiles[VM_NAME]

    print(f"Selected technique: {vm_info['selected_technique']}")
    print(f"CPU limit: {vm_info['cpu_limit']}")
    print(f"Memory limit: {vm_info['memory_limit_mb']} MB")

    x, true_label, slice_path = load_sample_slice(SLICES_METADATA_PATH)

    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)
    state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item())

    print(f"Slice path: {slice_path}")
    print(f"True label: {true_label}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.6f}")
    print("Container inference finished.")


if __name__ == "__main__":
    main()