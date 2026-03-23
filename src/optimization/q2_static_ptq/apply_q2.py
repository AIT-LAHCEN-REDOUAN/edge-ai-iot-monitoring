from pathlib import Path
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = False
CALIBRATION_BATCHES = 20

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
Q2_MODEL_PATH = Path(r"D:\github\IOT_Project\models\optimized\q2_static_ptq_fx_model.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "q2_static_ptq_fx_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def calibrate_model(model, loader, num_batches=20):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Calibrating Q2 FX", leave=True)):
            if batch_idx >= num_batches:
                break
            images = images.to(DEVICE)
            _ = model(images)


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Q2 FX accuracy", leave=True):
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
        for images, _ in tqdm(loader, desc="Benchmarking Q2 FX inference", leave=True):
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
    print("⚙️ APPLYING Q2: STATIC PTQ WITH FX GRAPH MODE")
    print(f"Using device: {DEVICE}")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    Q2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    calibration_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Load baseline model
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False)
    state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # FX quantization requires example input
    example_inputs = (torch.randn(1, 1, 128, 128),)

    # QConfig mapping
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    # Prepare FX
    prepared_model = prepare_fx(
        copy.deepcopy(model),
        qconfig_mapping=qconfig_mapping,
        example_inputs=example_inputs
    )

    # Calibrate
    calibrate_model(prepared_model, calibration_loader, num_batches=CALIBRATION_BATCHES)

    # Convert FX
    quantized_model = convert_fx(prepared_model)
    quantized_model.to(DEVICE)
    quantized_model.eval()

    # Save quantized model state_dict
    torch.save(quantized_model.state_dict(), Q2_MODEL_PATH)

    # Metrics
    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    q2_size_mb = get_file_size_mb(Q2_MODEL_PATH)

    test_acc = evaluate_accuracy(quantized_model, test_loader)
    avg_batch_time, avg_image_time, throughput = benchmark_inference(quantized_model, test_loader)

    report = (
        "=== Q2 STATIC PTQ FX REPORT ===\n\n"
        f"Device: {DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"Q2 FX model path: {Q2_MODEL_PATH}\n\n"
        f"Baseline model size (MB): {baseline_size_mb:.4f}\n"
        f"Q2 FX model size (MB): {q2_size_mb:.4f}\n"
        f"Size reduction (MB): {baseline_size_mb - q2_size_mb:.4f}\n"
        f"Compression ratio: {baseline_size_mb / q2_size_mb:.4f}\n\n"
        f"Q2 FX test accuracy: {test_acc:.6f}\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ Q2 STATIC PTQ FX COMPLETED")
    print("==============================")
    print(f"Baseline model size (MB): {baseline_size_mb:.4f}")
    print(f"Q2 FX model size (MB): {q2_size_mb:.4f}")
    print(f"Q2 FX test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()