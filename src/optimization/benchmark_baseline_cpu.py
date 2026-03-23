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

MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
OUTPUT_TXT = RESULTS_DIR / "baseline_benchmark_cpu.txt"


# ===============================
# HELPERS
# ===============================
def get_model_size_mb(model_path: Path) -> float:
    return model_path.stat().st_size / (1024 * 1024)


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating CPU baseline accuracy", leave=True):
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
        for images, _ in tqdm(loader, desc="Benchmarking CPU baseline inference", leave=True):
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


def save_report(text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


# ===============================
# MAIN
# ===============================
def main():
    print("📏 BENCHMARKING BASELINE MODEL ON CPU")
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
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    model_size_mb = get_model_size_mb(MODEL_PATH)
    test_acc = evaluate_accuracy(model, test_loader)
    avg_batch_time, avg_image_time, throughput = benchmark_inference(model, test_loader)

    report_text = (
        "=== BASELINE CPU BENCHMARK ===\n\n"
        f"Device: {DEVICE}\n"
        f"Model path: {MODEL_PATH}\n"
        f"Model size (MB): {model_size_mb:.4f}\n"
        f"Test accuracy: {test_acc:.6f}\n\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ CPU BASELINE BENCHMARK COMPLETED")
    print("==============================")
    print(f"Model size (MB): {model_size_mb:.4f}")
    print(f"Test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")

    save_report(report_text, OUTPUT_TXT)
    print(f"\nBenchmark report saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()