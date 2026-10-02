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
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
Q5_FP16_CHECKPOINT_PATH = Path(r"D:\github\IOT_Project\models\optimized\q5_mixed_precision_fp16_checkpoint.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "q5_mixed_precision_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_fp16_checkpoint(fp32_state_dict, output_path: Path):
    fp16_state_dict = {}
    for key, value in fp32_state_dict.items():
        if torch.is_floating_point(value):
            fp16_state_dict[key] = value.half()
        else:
            fp16_state_dict[key] = value
    torch.save(fp16_state_dict, output_path)


def evaluate_accuracy_fp32(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Q5 FP32 accuracy", leave=True):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_accuracy_amp(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Q5 AMP accuracy", leave=True):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_fp32(model, loader):
    model.eval()
    batch_times = []
    total_images = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Benchmarking Q5 FP32 inference", leave=True):
            images = images.to(DEVICE, non_blocking=True)

            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(images)

            torch.cuda.synchronize()
            end = time.perf_counter()

            batch_times.append(end - start)
            total_images += images.size(0)

    avg_batch_time = float(np.mean(batch_times))
    avg_image_time = avg_batch_time / loader.batch_size
    throughput = total_images / sum(batch_times)
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return avg_batch_time, avg_image_time, throughput, peak_gpu_memory_mb


def benchmark_amp(model, loader):
    model.eval()
    batch_times = []
    total_images = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Benchmarking Q5 AMP inference", leave=True):
            images = images.to(DEVICE, non_blocking=True)

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(images)

            torch.cuda.synchronize()
            end = time.perf_counter()

            batch_times.append(end - start)
            total_images += images.size(0)

    avg_batch_time = float(np.mean(batch_times))
    avg_image_time = avg_batch_time / loader.batch_size
    throughput = total_images / sum(batch_times)
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return avg_batch_time, avg_image_time, throughput, peak_gpu_memory_mb


def save_report(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ===============================
# MAIN
# ===============================
def main():
    print("⚙️ APPLYING Q5: MIXED PRECISION")
    print(f"Using device: {DEVICE}")

    if DEVICE.type != "cuda":
        raise RuntimeError("Q5 mixed precision requires CUDA/GPU to be meaningful.")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    Q5_FP16_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)
    baseline_state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(baseline_state_dict)
    model.eval()

    # Save FP16 checkpoint for reference
    save_fp16_checkpoint(baseline_state_dict, Q5_FP16_CHECKPOINT_PATH)

    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    q5_fp16_checkpoint_size_mb = get_file_size_mb(Q5_FP16_CHECKPOINT_PATH)

    # Accuracy
    fp32_acc = evaluate_accuracy_fp32(model, test_loader)
    amp_acc = evaluate_accuracy_amp(model, test_loader)

    # Benchmark
    fp32_batch_time, fp32_image_time, fp32_throughput, fp32_peak_mem = benchmark_fp32(model, test_loader)
    amp_batch_time, amp_image_time, amp_throughput, amp_peak_mem = benchmark_amp(model, test_loader)

    report = (
        "=== Q5 MIXED PRECISION REPORT ===\n\n"
        f"Device: {DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"Q5 FP16 checkpoint path: {Q5_FP16_CHECKPOINT_PATH}\n\n"
        f"Baseline checkpoint size (MB): {baseline_size_mb:.4f}\n"
        f"Q5 FP16 checkpoint size (MB): {q5_fp16_checkpoint_size_mb:.4f}\n"
        f"Checkpoint size reduction (MB): {baseline_size_mb - q5_fp16_checkpoint_size_mb:.4f}\n"
        f"Checkpoint compression ratio: {baseline_size_mb / q5_fp16_checkpoint_size_mb:.4f}\n\n"
        f"FP32 test accuracy: {fp32_acc:.6f}\n"
        f"AMP test accuracy: {amp_acc:.6f}\n\n"
        f"FP32 avg batch inference time (sec): {fp32_batch_time:.6f}\n"
        f"FP32 avg image inference time (sec): {fp32_image_time:.6f}\n"
        f"FP32 throughput (images/sec): {fp32_throughput:.2f}\n"
        f"FP32 peak GPU memory (MB): {fp32_peak_mem:.2f}\n\n"
        f"AMP avg batch inference time (sec): {amp_batch_time:.6f}\n"
        f"AMP avg image inference time (sec): {amp_image_time:.6f}\n"
        f"AMP throughput (images/sec): {amp_throughput:.2f}\n"
        f"AMP peak GPU memory (MB): {amp_peak_mem:.2f}\n"
    )

    print("\n==============================")
    print("✅ Q5 MIXED PRECISION COMPLETED")
    print("==============================")
    print(f"Baseline checkpoint size (MB): {baseline_size_mb:.4f}")
    print(f"Q5 FP16 checkpoint size (MB): {q5_fp16_checkpoint_size_mb:.4f}")
    print(f"FP32 test accuracy: {fp32_acc:.6f}")
    print(f"AMP test accuracy: {amp_acc:.6f}")

    print("\n--- FP32 GPU Benchmark ---")
    print(f"Avg batch inference time (sec): {fp32_batch_time:.6f}")
    print(f"Avg image inference time (sec): {fp32_image_time:.6f}")
    print(f"Throughput (images/sec): {fp32_throughput:.2f}")
    print(f"Peak GPU memory (MB): {fp32_peak_mem:.2f}")

    print("\n--- AMP GPU Benchmark ---")
    print(f"Avg batch inference time (sec): {amp_batch_time:.6f}")
    print(f"Avg image inference time (sec): {amp_image_time:.6f}")
    print(f"Throughput (images/sec): {amp_throughput:.2f}")
    print(f"Peak GPU memory (MB): {amp_peak_mem:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()