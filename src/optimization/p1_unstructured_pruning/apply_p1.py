from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm

from src.dataset.dataloader import create_dataloader, DEFAULT_METADATA_PATH
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DEVICE = torch.device("cpu")

BATCH_SIZE = 64
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False

PRUNE_AMOUNT = 0.30
FINETUNE_EPOCHS = 2
LEARNING_RATE = 1e-4

BASELINE_MODEL_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
P1_MODEL_PATH = Path(r"D:\github\IOT_Project\models\optimized\p1_unstructured_pruned_model.pth")

RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
REPORT_PATH = RESULTS_DIR / "p1_unstructured_pruning_report.txt"


# ===============================
# HELPERS
# ===============================
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def count_zero_weights(model):
    zero_count = 0
    total_count = 0

    for param in model.parameters():
        if param is not None:
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()

    sparsity = zero_count / total_count if total_count > 0 else 0.0
    return zero_count, total_count, sparsity


def apply_global_unstructured_pruning(model, amount=0.3):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return parameters_to_prune


def remove_pruning_reparameterization(parameters_to_prune):
    for module, name in parameters_to_prune:
        prune.remove(module, name)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="P1 Fine-tuning", leave=True)

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


def evaluate_accuracy(model, loader, device, desc="Evaluating P1 accuracy"):
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
        for images, _ in tqdm(loader, desc="Benchmarking P1 inference", leave=True):
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
    print("✂️ APPLYING P1: UNSTRUCTURED PRUNING")
    print(f"Training device: {TRAIN_DEVICE}")
    print(f"Evaluation device: {EVAL_DEVICE}")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_MODEL_PATH}")

    P1_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    # 1) Load baseline
    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False)
    baseline_state_dict = torch.load(BASELINE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(baseline_state_dict)

    # 2) Apply pruning
    parameters_to_prune = apply_global_unstructured_pruning(model, amount=PRUNE_AMOUNT)

    # 3) Fine-tune pruned model on GPU
    model = model.to(TRAIN_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_train_start = time.time()

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\n📚 P1 Fine-tuning Epoch {epoch + 1}/{FINETUNE_EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=TRAIN_DEVICE
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    total_train_time = time.time() - total_train_start

    # 4) Remove pruning wrappers so weights become standard tensors with zeros
    model = model.to("cpu")
    remove_pruning_reparameterization(parameters_to_prune)
    model.eval()

    # 5) Save
    torch.save(model.state_dict(), P1_MODEL_PATH)

    # 6) Evaluate and benchmark on CPU
    baseline_size_mb = get_file_size_mb(BASELINE_MODEL_PATH)
    p1_size_mb = get_file_size_mb(P1_MODEL_PATH)

    zero_count, total_count, sparsity = count_zero_weights(model)

    test_acc = evaluate_accuracy(
        model,
        test_loader_cpu,
        device=EVAL_DEVICE,
        desc="Evaluating P1 pruned accuracy"
    )

    avg_batch_time, avg_image_time, throughput = benchmark_inference(
        model,
        test_loader_cpu,
        device=EVAL_DEVICE
    )

    report = (
        "=== P1 UNSTRUCTURED PRUNING REPORT ===\n\n"
        f"Training device: {TRAIN_DEVICE}\n"
        f"Evaluation device: {EVAL_DEVICE}\n"
        f"Baseline model path: {BASELINE_MODEL_PATH}\n"
        f"P1 model path: {P1_MODEL_PATH}\n\n"
        f"Baseline model size (MB): {baseline_size_mb:.4f}\n"
        f"P1 model size (MB): {p1_size_mb:.4f}\n"
        f"Size reduction (MB): {baseline_size_mb - p1_size_mb:.4f}\n"
        f"Compression ratio: {baseline_size_mb / p1_size_mb:.4f}\n\n"
        f"Prune amount target: {PRUNE_AMOUNT:.2f}\n"
        f"Zero weights: {zero_count}\n"
        f"Total weights: {total_count}\n"
        f"Measured sparsity: {sparsity:.6f}\n\n"
        f"Fine-tuning epochs: {FINETUNE_EPOCHS}\n"
        f"Fine-tuning time (sec): {total_train_time:.2f}\n\n"
        f"P1 test accuracy: {test_acc:.6f}\n"
        f"Average batch inference time (sec): {avg_batch_time:.6f}\n"
        f"Average image inference time (sec): {avg_image_time:.6f}\n"
        f"Throughput (images/sec): {throughput:.2f}\n"
    )

    print("\n==============================")
    print("✅ P1 UNSTRUCTURED PRUNING COMPLETED")
    print("==============================")
    print(f"Baseline model size (MB): {baseline_size_mb:.4f}")
    print(f"P1 model size (MB): {p1_size_mb:.4f}")
    print(f"Measured sparsity: {sparsity:.6f}")
    print(f"P1 test accuracy: {test_acc:.6f}")
    print(f"Average batch inference time (sec): {avg_batch_time:.6f}")
    print(f"Average image inference time (sec): {avg_image_time:.6f}")
    print(f"Throughput (images/sec): {throughput:.2f}")
    print(f"Fine-tuning time (sec): {total_train_time:.2f}")

    save_report(report, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()