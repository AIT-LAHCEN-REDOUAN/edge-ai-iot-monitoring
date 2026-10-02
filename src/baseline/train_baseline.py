from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset.dataloader import create_all_dataloaders
from src.baseline.resnet18_model import ResNet18BinaryClassifier


# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

MODEL_SAVE_PATH = Path(r"D:\github\IOT_Project\models\baseline\best_model.pth")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ===============================
# TRAIN FUNCTION
# ===============================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ===============================
# VALIDATION FUNCTION
# ===============================
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}"
            )

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ===============================
# MAIN
# ===============================
def main():
    print("🚀 TRAINING BASELINE MODEL")
    print(f"Using device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    train_loader, val_loader, _ = create_all_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = ResNet18BinaryClassifier(num_classes=2, pretrained=False).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    total_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion
        )

        epoch_time = time.time() - epoch_start_time

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} sec")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Best model saved!")

    total_time = time.time() - total_start_time

    print("\n==============================")
    print("✅ TRAINING COMPLETED")
    print("==============================")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Total training time: {total_time:.2f} sec")


if __name__ == "__main__":
    main()