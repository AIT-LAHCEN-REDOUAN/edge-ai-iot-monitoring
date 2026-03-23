from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.ao.quantization import get_default_qat_qconfig

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.optimization.q3_qat.qat_resnet18_model import (
    QuantizableResNet18,
    load_baseline_weights_into_qat_model
)


# ===============================
# CONFIG
# ===============================
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DEVICE = torch.device("cpu")   # quantized inference happens on CPU

BATCH_SIZE = 64
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False

QAT_EPOCHS = 2
LEARNING_RATE = 1e-4

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
Q3_MODEL_PATH = Path(r"D:\github\IOT_Project\models\optimized\q3_qat_model.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "q3_qat_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="QAT Training", leave=True)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    return total_loss / len(loader), correct / total


def evaluate_accuracy(model, loader, device, desc="Evaluating Q3 accuracy"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=True):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_inference(model, loader, device):
    model.eval()
    batch_times = []
    total_images = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Benchmarking Q3 inference", leave=True):
            images = images.to(device)

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
    print("⚙️ APPLYING Q3: FAST QAT")
    print(f"Training device: {TRAIN_DEVICE}")
    print(f"Evaluation device: {EVAL_DEVICE}")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    Q3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader_cpu = create_dataloader(
        metadata_csv=DEFAULT_METADATA_PATH,
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 1) Create QAT-ready model
    model = QuantizableResNet18(num_classes=2, pretrained=False)

    baseline_state_dict = torch.load(
        BASELINE_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    missing, unexpected = load_baseline_weights_into_qat_model(model, baseline_state_dict)

    print("\nBaseline weights loaded into QAT model.")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    # 2) Put model in eval mode BEFORE fusion
    model.eval()
    model.fuse_model()

    # 3) Set QAT config
    model.qconfig = get_default_qat_qconfig("fbgemm")

    # 4) Switch to train mode BEFORE prepare_qat
    model.train()
    model = torch.ao.quantization.prepare_qat(model, inplace=False)

    # 5) IMPORTANT: move prepared QAT model to GPU AFTER prepare_qat
    model = model.to(TRAIN_DEVICE)

    # 6) Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7) Fast fine-tuning
    total_train_start = time.time()

    for epoch in range(QAT_EPOCHS):
        print(f"\n📚 QAT Epoch {epoch + 1}/{QAT_EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=TRAIN_DEVICE
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    total_train_time = time.time() - total_train_start

    # 8) Convert to quantized CPU model
    model = model.to("cpu")
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    quantized_model.eval()

    # 9) Save
    torch.save(quantized_model.state_dict(), Q3_MODEL_PATH)

    # 10) Evaluate and benchmark on CPU
    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    q3_size_mb = get_file_size_mb(Q3_MODEL_PATH)

    test_acc = evaluate_accuracy(
        quantized_model,
        test_loader_cpu,
        device=EVAL_DEVICE,
        desc="Evaluating Q3 quantized accuracy"
    )

    avg_batch_time, avg_image_time, throughput = benchmark_inference(
        quantized_model,
        test_loader_cpu,
        device=EVAL_DEVICE
    )

    report = (
        "=== Q3 FAST QAT REPORT ===\n\n"
        f"Training device: {TRAIN_DEVICE}\n"
        f"Evaluation device: {EVAL_DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"Q3 model path: {Q3_MODEL_PATH}\n\n"
        f"Baseline model size (MB): {baseline_size_mb:.4f}\n"
        f"Q3 model size (MB): {q3_size_mb:.4f}\n"
        f"Size reduction (MB): {baseline_size_mb - q3_size_mb:.4f}\n"
        f"Compression ratio: {baseline_size_mb / q3_size_mb:.4f}\n\n"
        f"QAT fine-tuning epochs: {QAT_EPOCHS}\n"
        f"QAT total training time (sec): {total_train_time:.2f}\n\n"
        f"Q3 test accuracy: {test_acc:.6f}\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ Q3 FAST QAT COMPLETED")
    print("==============================")
    print(f"Baseline model size (MB): {baseline_size_mb:.4f}")
    print(f"Q3 model size (MB): {q3_size_mb:.4f}")
    print(f"Q3 test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")
    print(f"QAT total training time (sec): {total_train_time:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()