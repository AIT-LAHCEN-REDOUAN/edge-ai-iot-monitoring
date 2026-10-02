from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.baseline.resnet18_model import ResNet18BinaryClassifier

# ===============================
# CONFIG
# ===============================
VM_PROFILES_PATH = Path(r"D:\github\IOT_Project\results\deployment\vm_profiles.json")
SLICES_METADATA_PATH = Path(r"D:\github\IOT_Project\data\metadata\processed_slices_metadata.csv")

MODELS_DIR = Path(r"D:\github\IOT_Project\models\optimized")
BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")

DEVICE = torch.device("cpu")

TECHNIQUE_TO_MODEL_PATH = {
    "q3_qat": MODELS_DIR / "q3_qat_model.pth",
    "q2_static_ptq_fx": MODELS_DIR / "q2_static_ptq_fx_model.pth",
    "q4_weight_only": MODELS_DIR / "q4_weight_only_fp16_model.pth",
    "baseline": BASELINE_MODEL_PATH,
}

HISTORICAL_WEIGHTS = {
    "q3_qat": 0.990530,
    "q2_static_ptq_fx": 0.987013,
    "q4_weight_only": 0.987013,
    "baseline": 0.987013,
}


# ===============================
# HELPERS
# ===============================
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
        raise ValueError("No successful test slices found in metadata.")

    sample_row = df.iloc[0]
    slice_path = Path(sample_row["slice_filepath"])
    label = int(sample_row["label"])

    if not slice_path.exists():
        raise FileNotFoundError(f"Sample slice file not found: {slice_path}")

    image = np.load(slice_path).astype(np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {image.shape}")

    image = np.expand_dims(image, axis=0)   # (1, H, W)
    image = np.expand_dims(image, axis=0)   # (1, 1, H, W)

    tensor = torch.from_numpy(image).float().to(DEVICE)
    return tensor, label, str(slice_path)


def build_model_for_technique(technique: str):
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)

    if technique == "q4_weight_only":
        state_dict = torch.load(TECHNIQUE_TO_MODEL_PATH[technique], map_location="cpu", weights_only=True)
        fp32_state_dict = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                fp32_state_dict[key] = value.float()
            else:
                fp32_state_dict[key] = value
        model.load_state_dict(fp32_state_dict)

    elif technique == "baseline":
        state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    elif technique == "q3_qat":
        # Q3 saved quantized state_dict is not directly loadable into plain ResNet18.
        # For orchestration simulation, use the baseline-compatible behavior with Q3 historical score.
        state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    elif technique == "q2_static_ptq_fx":
        # FX quantized model also not directly loadable into plain ResNet18.
        # For orchestration simulation, use the baseline-compatible behavior with Q2 historical score.
        state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    else:
        raise ValueError(f"Unsupported technique: {technique}")

    model.eval()
    return model


def run_node_inference(vm_name: str, technique: str, input_tensor: torch.Tensor):
    model = build_model_for_technique(technique)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

    historical_weight = HISTORICAL_WEIGHTS.get(technique, 0.5)
    vote_weight = historical_weight * confidence

    return {
        "vm_name": vm_name,
        "technique": technique,
        "prediction": pred,
        "confidence": confidence,
        "historical_weight": historical_weight,
        "vote_weight": vote_weight,
    }


def weighted_voting(node_outputs):
    class_scores = {}

    for output in node_outputs:
        pred = output["prediction"]
        weight = output["vote_weight"]

        if pred not in class_scores:
            class_scores[pred] = 0.0

        class_scores[pred] += weight

    final_prediction = max(class_scores, key=class_scores.get)
    total_weight = sum(class_scores.values())
    final_confidence = class_scores[final_prediction] / total_weight if total_weight > 0 else 0.0

    return final_prediction, final_confidence, class_scores


# ===============================
# MAIN
# ===============================
def main():
    print("🧠 RUNNING COLLECTIVE ORCHESTRATOR...\n")

    vm_profiles = load_vm_profiles(VM_PROFILES_PATH)
    input_tensor, true_label, sample_path = load_sample_slice(SLICES_METADATA_PATH)

    print(f"Sample slice: {sample_path}")
    print(f"True label: {true_label}\n")

    node_outputs = []

    for vm_name, vm_info in vm_profiles.items():
        technique = vm_info["selected_technique"]

        print(f"Running {vm_name} with technique: {technique}")
        result = run_node_inference(vm_name, technique, input_tensor)
        node_outputs.append(result)

        print(
            f" -> prediction={result['prediction']} | "
            f"confidence={result['confidence']:.6f} | "
            f"historical_weight={result['historical_weight']:.6f} | "
            f"vote_weight={result['vote_weight']:.6f}"
        )

    final_prediction, final_confidence, class_scores = weighted_voting(node_outputs)

    print("\n==============================")
    print("✅ COLLECTIVE DECISION COMPLETED")
    print("==============================")
    print(f"Final prediction: {final_prediction}")
    print(f"Final confidence: {final_confidence:.6f}")
    print(f"True label: {true_label}")
    print(f"Class scores: {class_scores}")

    if final_prediction == true_label:
        print("Collective result: CORRECT")
    else:
        print("Collective result: INCORRECT")


if __name__ == "__main__":
    main()