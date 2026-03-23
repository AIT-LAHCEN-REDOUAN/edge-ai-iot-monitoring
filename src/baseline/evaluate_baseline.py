from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
RESULTS_DIR = Path(r"D:\github\IOT_Project\results\baseline")
METRICS_TXT_PATH = RESULTS_DIR / "metrics.txt"
CONFUSION_MATRIX_IMG_PATH = RESULTS_DIR / "confusion_matrix.png"

CLASS_NAMES = ["Class_0", "Class_1"]


# ===============================
# HELPERS
# ===============================
def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_preds = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def save_metrics_to_file(metrics_text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)


def save_confusion_matrix(cm, class_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===============================
# MAIN
# ===============================
def main():
    print("📊 EVALUATING BASELINE MODEL ON TEST SET")
    print(f"Using device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    y_true, y_pred = evaluate_model(model, test_loader)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    metrics_text = (
        "=== BASELINE TEST METRICS ===\n\n"
        f"Accuracy : {acc:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall   : {recall:.6f}\n"
        f"F1-Score : {f1:.6f}\n\n"
        "=== CONFUSION MATRIX ===\n"
        f"{cm}\n\n"
        "=== CLASSIFICATION REPORT ===\n"
        f"{report}\n"
    )

    print("\n==============================")
    print("✅ TEST EVALUATION COMPLETED")
    print("==============================")
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1-Score : {f1:.6f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    save_metrics_to_file(metrics_text, METRICS_TXT_PATH)
    save_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_IMG_PATH)

    print(f"Metrics saved to: {METRICS_TXT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_IMG_PATH}")


if __name__ == "__main__":
    main()