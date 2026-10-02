from pathlib import Path
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cpu")  # dynamic quantization is mainly for CPU
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = False

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
QUANTIZED_MODEL_PATH = Path(r"D:\github\IOT_Project\models\optimized\q1_dynamic_quantized_model.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "q1_dynamic_quantization_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Q1 accuracy", leave=True):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_inference(model, loader):
    model.eval()

    batch_times = []
    total_images = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Benchmarking Q1 inference", leave=True):
            images = images.to(DEVICE)

            start = time.perf_counter()
            _ = model(images)
            end = time.perf_counter()

            batch_times.append(end - start)
            total_images += images.size(0)

    avg_batch_time = float(np.mean(batch_times))
    avg_image_time = avg_batch_time / loader.batch_size
    throughput = total_images / sum(batch_times)

    return avg_batch_time, avg_image_time, throughput


def save_report(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ===============================
# MAIN
# ===============================
def main():
    print("⚙️ APPLYING Q1: DYNAMIC QUANTIZATION")
    print(f"Using device: {DEVICE}")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    QUANTIZED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Load baseline model
    baseline_model = ResNet18BinaryClassifier(num_classes=2, pretrained=False)
    state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    baseline_model.load_state_dict(state_dict)
    baseline_model.eval()

    # Apply dynamic quantization
    quantized_model = copy.deepcopy(baseline_model)
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    quantized_model.to(DEVICE)
    quantized_model.eval()

    # Save quantized state_dict
    torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

    # Metrics
    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    quantized_size_mb = get_file_size_mb(QUANTIZED_MODEL_PATH)

    test_acc = evaluate_accuracy(quantized_model, test_loader)
    avg_batch_time, avg_image_time, throughput = benchmark_inference(quantized_model, test_loader)

    report = (
        "=== Q1 DYNAMIC QUANTIZATION REPORT ===\n\n"
        f"Device: {DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"Quantized model path: {QUANTIZED_MODEL_PATH}\n\n"
        f"Baseline model size (MB): {baseline_size_mb:.4f}\n"
        f"Quantized model size (MB): {quantized_size_mb:.4f}\n"
        f"Size reduction (MB): {baseline_size_mb - quantized_size_mb:.4f}\n"
        f"Compression ratio: {baseline_size_mb / quantized_size_mb:.4f}\n\n"
        f"Q1 test accuracy: {test_acc:.6f}\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ Q1 DYNAMIC QUANTIZATION COMPLETED")
    print("==============================")
    print(f"Baseline model size (MB): {baseline_size_mb:.4f}")
    print(f"Quantized model size (MB): {quantized_size_mb:.4f}")
    print(f"Q1 test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()