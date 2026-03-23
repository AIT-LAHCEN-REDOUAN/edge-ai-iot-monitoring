from pathlib import Path
import time
import torch
import numpy as np
from tqdm import tqdm

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cpu")

BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = False

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
Q4_MODEL_PATH = Path(r"D:\github\IOT_Project\models\optimized\q4_weight_only_fp16_model.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "q4_weight_only_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_fp16_state_dict(fp32_state_dict, output_path: Path):
    fp16_state_dict = {}
    for key, value in fp32_state_dict.items():
        if torch.is_floating_point(value):
            fp16_state_dict[key] = value.half()
        else:
            fp16_state_dict[key] = value
    torch.save(fp16_state_dict, output_path)


def load_fp16_weights_into_fp32_model(model, fp16_path: Path):
    state_dict = torch.load(fp16_path, map_location="cpu", weights_only=True)
    fp32_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            fp32_state_dict[key] = value.float()
        else:
            fp32_state_dict[key] = value
    model.load_state_dict(fp32_state_dict)
    return model


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Q4 accuracy", leave=True):
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
        for images, _ in tqdm(loader, desc="Benchmarking Q4 inference", leave=True):
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
    print("⚙️ APPLYING Q4: WEIGHT-ONLY QUANTIZATION (FP16 WEIGHT STORAGE)")
    print(f"Using device: {DEVICE}")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    Q4_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Load baseline weights
    baseline_state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)

    # Save compressed weight-only version
    save_fp16_state_dict(baseline_state_dict, Q4_MODEL_PATH)

    # Rebuild model for evaluation
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)
    model = load_fp16_weights_into_fp32_model(model, Q4_MODEL_PATH)
    model.eval()

    # Metrics
    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    q4_size_mb = get_file_size_mb(Q4_MODEL_PATH)

    test_acc = evaluate_accuracy(model, test_loader)
    avg_batch_time, avg_image_time, throughput = benchmark_inference(model, test_loader)

    report = (
        "=== Q4 WEIGHT-ONLY QUANTIZATION REPORT ===\n\n"
        f"Device: {DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"Q4 model path: {Q4_MODEL_PATH}\n\n"
        f"Baseline model size (MB): {baseline_size_mb:.4f}\n"
        f"Q4 model size (MB): {q4_size_mb:.4f}\n"
        f"Size reduction (MB): {baseline_size_mb - q4_size_mb:.4f}\n"
        f"Compression ratio: {baseline_size_mb / q4_size_mb:.4f}\n\n"
        f"Q4 test accuracy: {test_acc:.6f}\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ Q4 WEIGHT-ONLY QUANTIZATION COMPLETED")
    print("==============================")
    print(f"Baseline model size (MB): {baseline_size_mb:.4f}")
    print(f"Q4 model size (MB): {q4_size_mb:.4f}")
    print(f"Q4 test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()