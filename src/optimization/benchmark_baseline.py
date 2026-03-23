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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
OUTPUT_TXT = RESULTS_DIR / "baseline_benchmark.txt"


# ===============================
# HELPERS
# ===============================
def get_model_size_mb(model_path: Path) -> float:
    size_bytes = model_path.stat().st_size
    return size_bytes / (1024 * 1024)


def evaluate_accuracy(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating accuracy", leave=True):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_inference(model, loader):
    model.eval()

    batch_times = []
    total_images = 0

    # Reset GPU memory stats if CUDA is available
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Benchmarking inference", leave=True):
            images = images.to(DEVICE, non_blocking=True)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_images += images.size(0)

    avg_batch_time = float(np.mean(batch_times))
    avg_image_time = avg_batch_time / loader.batch_size
    throughput = total_images / sum(batch_times)

    peak_gpu_memory_mb = None
    if DEVICE.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "avg_batch_time_sec": avg_batch_time,
        "avg_image_time_sec": avg_image_time,
        "throughput_img_per_sec": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb
    }


def save_report(report_text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)


# ===============================
# MAIN
# ===============================
def main():
    print("📏 BENCHMARKING BASELINE MODEL")
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

    # Model size
    model_size_mb = get_model_size_mb(MODEL_PATH)

    # Accuracy
    test_acc = evaluate_accuracy(model, test_loader)

    # Inference benchmark
    benchmark_results = benchmark_inference(model, test_loader)

    report_text = (
        "=== BASELINE MODEL BENCHMARK ===\n\n"
        f"Device: {DEVICE}\n"
        f"Model path: {MODEL_PATH}\n"
        f"Model size (MB): {model_size_mb:.4f}\n"
        f"Test accuracy: {test_acc:.6f}\n\n"
        f"Average batch inference time (sec): {benchmark_results['avg_batch_time_sec']:.6f}\n"
        f"Average image inference time (sec): {benchmark_results['avg_image_time_sec']:.6f}\n"
        f"Throughput (images/sec): {benchmark_results['throughput_img_per_sec']:.2f}\n"
    )

    if benchmark_results["peak_gpu_memory_mb"] is not None:
        report_text += f"Peak GPU memory (MB): {benchmark_results['peak_gpu_memory_mb']:.2f}\n"

    print("\n==============================")
    print("✅ BASELINE BENCHMARK COMPLETED")
    print("==============================")
    print(f"Model size (MB): {model_size_mb:.4f}")
    print(f"Test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {benchmark_results['avg_batch_time_sec']:.6f}")
    print(f"Average image inference time (sec): {benchmark_results['avg_image_time_sec']:.6f}")
    print(f"Throughput (images/sec): {benchmark_results['throughput_img_per_sec']:.2f}")

    if benchmark_results["peak_gpu_memory_mb"] is not None:
        print(f"Peak GPU memory (MB): {benchmark_results['peak_gpu_memory_mb']:.2f}")

    save_report(report_text, OUTPUT_TXT)
    print(f"\nBenchmark report saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()