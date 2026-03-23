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

CONFIDENCE_THRESHOLD = 0.90
REVALIDATION_NUM_EXTRA_SLICES = 5

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


def load_test_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Slices metadata file not found: {path}")

    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    df = df[df["split"] == "test"].copy()

    if df.empty:
        raise ValueError("No successful test slices found.")

    return df.reset_index(drop=True)


def load_slice_as_tensor(slice_path: Path):
    if not slice_path.exists():
        raise FileNotFoundError(f"Slice file not found: {slice_path}")

    image = np.load(slice_path).astype(np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {image.shape}")

    image = np.expand_dims(image, axis=0)   # (1, H, W)
    image = np.expand_dims(image, axis=0)   # (1, 1, H, W)

    return torch.from_numpy(image).float().to(DEVICE)


def select_first_pass_sample(df: pd.DataFrame):
    row = df.iloc[0]
    return {
        "subject_id": str(row["subject_id"]),
        "slice_path": Path(row["slice_filepath"]),
        "true_label": int(row["label"]),
        "slice_index": int(row["slice_index"]),
    }


def select_revalidation_slices(df: pd.DataFrame, subject_id: str, exclude_slice_index: int, k: int):
    subject_df = df[df["subject_id"] == subject_id].copy()
    subject_df = subject_df[subject_df["slice_index"] != exclude_slice_index].copy()
    subject_df = subject_df.sort_values(by="slice_index").reset_index(drop=True)

    if subject_df.empty:
        return []

    center_slice = exclude_slice_index

    subject_df["distance"] = (subject_df["slice_index"] - center_slice).abs()
    subject_df = subject_df.sort_values(by="distance").reset_index(drop=True)

    selected = subject_df.head(k)
    return [
        {
            "subject_id": str(row["subject_id"]),
            "slice_path": Path(row["slice_filepath"]),
            "true_label": int(row["label"]),
            "slice_index": int(row["slice_index"]),
        }
        for _, row in selected.iterrows()
    ]


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
        # simulation-compatible fallback
        state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    elif technique == "q2_static_ptq_fx":
        # simulation-compatible fallback
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


def run_collective_round(vm_profiles, input_tensor):
    node_outputs = []

    for vm_name, vm_info in vm_profiles.items():
        technique = vm_info["selected_technique"]
        result = run_node_inference(vm_name, technique, input_tensor)
        node_outputs.append(result)

    final_prediction, final_confidence, class_scores = weighted_voting(node_outputs)

    return {
        "node_outputs": node_outputs,
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "class_scores": class_scores,
    }


# ===============================
# MAIN
# ===============================
def main():
    print("🔁 RUNNING CONFIDENCE-BASED REVALIDATION...\n")

    vm_profiles = load_vm_profiles(VM_PROFILES_PATH)
    df = load_test_metadata(SLICES_METADATA_PATH)

    # -------------------------------
    # First pass
    # -------------------------------
    first_sample = select_first_pass_sample(df)
    first_tensor = load_slice_as_tensor(first_sample["slice_path"])

    print("FIRST PASS")
    print(f"Subject ID: {first_sample['subject_id']}")
    print(f"Slice path: {first_sample['slice_path']}")
    print(f"Slice index: {first_sample['slice_index']}")
    print(f"True label: {first_sample['true_label']}\n")

    first_pass = run_collective_round(vm_profiles, first_tensor)

    for output in first_pass["node_outputs"]:
        print(
            f"{output['vm_name']} -> pred={output['prediction']} | "
            f"conf={output['confidence']:.6f} | "
            f"vote_weight={output['vote_weight']:.6f}"
        )

    print("\nFirst-pass collective result:")
    print(f"Prediction: {first_pass['final_prediction']}")
    print(f"Confidence: {first_pass['final_confidence']:.6f}")
    print(f"Class scores: {first_pass['class_scores']}")

    # -------------------------------
    # Revalidation check
    # -------------------------------
    if first_pass["final_confidence"] >= CONFIDENCE_THRESHOLD:
        print("\n✅ Confidence is high enough. No revalidation needed.")
        print(f"Final prediction: {first_pass['final_prediction']}")
        print(f"True label: {first_sample['true_label']}")
        print(
            "Final result: "
            + ("CORRECT" if first_pass["final_prediction"] == first_sample["true_label"] else "INCORRECT")
        )
        return

    print("\n⚠️ Confidence below threshold. Starting revalidation...")

    extra_samples = select_revalidation_slices(
        df=df,
        subject_id=first_sample["subject_id"],
        exclude_slice_index=first_sample["slice_index"],
        k=REVALIDATION_NUM_EXTRA_SLICES
    )

    if not extra_samples:
        print("No extra slices available for revalidation.")
        print(f"Fallback final prediction: {first_pass['final_prediction']}")
        return

    all_second_pass_outputs = []

    print("\nSECOND PASS (additional slices)")
    for idx, sample in enumerate(extra_samples, start=1):
        print(f"\nExtra slice {idx}/{len(extra_samples)}")
        print(f" -> slice path: {sample['slice_path']}")
        print(f" -> slice index: {sample['slice_index']}")

        tensor = load_slice_as_tensor(sample["slice_path"])
        round_result = run_collective_round(vm_profiles, tensor)

        for output in round_result["node_outputs"]:
            print(
                f"    {output['vm_name']} -> pred={output['prediction']} | "
                f"conf={output['confidence']:.6f} | "
                f"vote_weight={output['vote_weight']:.6f}"
            )
            all_second_pass_outputs.append(output)

    # Aggregate revalidation outputs
    second_prediction, second_confidence, second_scores = weighted_voting(all_second_pass_outputs)

    print("\n==============================")
    print("✅ REVALIDATION COMPLETED")
    print("==============================")
    print(f"First-pass prediction: {first_pass['final_prediction']}")
    print(f"First-pass confidence: {first_pass['final_confidence']:.6f}")
    print(f"Revalidated prediction: {second_prediction}")
    print(f"Revalidated confidence: {second_confidence:.6f}")
    print(f"True label: {first_sample['true_label']}")
    print(f"Revalidated class scores: {second_scores}")

    print(
        "Revalidated result: "
        + ("CORRECT" if second_prediction == first_sample["true_label"] else "INCORRECT")
    )


if __name__ == "__main__":
    main()