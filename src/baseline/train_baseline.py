import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'nibabel'. Install it with: pip install nibabel"
    ) from exc


IMG_SIZE = 224
RANDOM_STATE = 42


@dataclass
class Sample:
    path: Path
    label: int
    slice_idx: int


class NiftiSliceDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int = IMG_SIZE, augment: bool = False) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self._cached_path: str | None = None
        self._cached_proxy = None

    def _get_proxy(self, path: Path):
        path_str = str(path)
        if self._cached_path != path_str or self._cached_proxy is None:
            nii = nib.load(path_str)
            self._cached_path = path_str
            self._cached_proxy = nii.dataobj
        return self._cached_proxy

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
        min_v = float(slice_2d.min())
        max_v = float(slice_2d.max())
        if max_v > min_v:
            return (slice_2d - min_v) / (max_v - min_v)
        return np.zeros_like(slice_2d, dtype=np.float32)

    @staticmethod
    def _read_slice_safe(proxy, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), proxy.shape[2] - 1))
        try:
            slice_2d = np.asanyarray(proxy[:, :, idx], dtype=np.float32)
        except Exception:
            fallback_idx = proxy.shape[2] // 2
            slice_2d = np.asanyarray(proxy[:, :, fallback_idx], dtype=np.float32)
        return slice_2d

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        proxy = self._get_proxy(sample.path)

        # Build a 2.5D input with neighboring slices as channels.
        indices = [sample.slice_idx - 1, sample.slice_idx, sample.slice_idx + 1]
        channels = [self._normalize_slice(self._read_slice_safe(proxy, idx)) for idx in indices]
        stacked = np.stack(channels, axis=0)

        tensor = torch.from_numpy(stacked).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if random.random() < 0.2:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() < 0.3:
                scale = 0.9 + 0.2 * random.random()
                bias = (random.random() - 0.5) * 0.1
                tensor = torch.clamp(tensor * scale + bias, 0.0, 1.0)

        return tensor, sample.label


def is_valid_nifti(file_path: Path) -> bool:
    try:
        nii = nib.load(str(file_path))
        proxy = nii.dataobj
        if len(proxy.shape) < 3:
            return False
        center_idx = proxy.shape[2] // 2
        _ = np.asanyarray(proxy[:, :, center_idx], dtype=np.float32)
        return True
    except Exception:
        return False


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_volumes(data_root: Path) -> Tuple[List[Tuple[Path, int, int]], List[str], int]:
    volumes: List[Tuple[Path, int, int]] = []
    class_names: List[str] = []
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    skipped_corrupted = 0

    for class_dir in class_dirs:
        nii_files = sorted(class_dir.rglob("*.nii")) + sorted(class_dir.rglob("*.nii.gz"))
        if not nii_files:
            continue
        idx = len(class_names)
        class_names.append(class_dir.name)
        for file_path in nii_files:
            try:
                nii = nib.load(str(file_path))
                proxy = nii.dataobj
                if len(proxy.shape) < 3 or proxy.shape[2] <= 0:
                    skipped_corrupted += 1
                    continue
                _ = np.asanyarray(proxy[:, :, proxy.shape[2] // 2], dtype=np.float32)
                volumes.append((file_path, idx, int(proxy.shape[2])))
            except Exception:
                skipped_corrupted += 1

    if len(class_names) < 2:
        raise ValueError(
            f"At least 2 class folders with NIfTI files are required in {data_root}. "
            f"Found: {class_names}"
        )

    if len(volumes) == 0:
        raise ValueError(f"No .nii or .nii.gz files found under {data_root}")

    if skipped_corrupted > 0:
        print(f"Skipped corrupted/unsupported NIfTI files: {skipped_corrupted}")

    return volumes, class_names, skipped_corrupted


def choose_slice_indices(depth: int, slices_per_volume: int) -> List[int]:
    if depth <= 1:
        return [0]

    start = max(0, int(depth * 0.10))
    end = min(depth - 1, int(depth * 0.90))
    if end <= start:
        start, end = 0, depth - 1

    if slices_per_volume <= 1:
        return [(start + end) // 2]

    lin = np.linspace(start, end, num=slices_per_volume)
    indices = sorted(set(int(round(x)) for x in lin))
    if not indices:
        indices = [depth // 2]
    return indices


def build_slice_samples(volumes: List[Tuple[Path, int, int]], slices_per_volume: int) -> List[Sample]:
    samples: List[Sample] = []
    for volume_path, label, depth in volumes:
        for slice_idx in choose_slice_indices(depth, slices_per_volume):
            samples.append(Sample(path=volume_path, label=label, slice_idx=slice_idx))
    return samples


def split_volumes(
    volumes: List[Tuple[Path, int, int]],
) -> Tuple[List[Tuple[Path, int, int]], List[Tuple[Path, int, int]], List[Tuple[Path, int, int]]]:
    labels = [v[1] for v in volumes]
    idx = list(range(len(volumes)))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    train_volumes = [volumes[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    test_volumes = [volumes[i] for i in test_idx]
    return train_volumes, val_volumes, test_volumes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)

    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_targets.extend(labels.numpy().tolist())

    return np.array(all_targets), np.array(all_preds)


@torch.no_grad()
def measure_inference(model: nn.Module, loader: DataLoader, device: torch.device, max_images: int = 100) -> Tuple[float, float]:
    model.eval()
    process = psutil.Process()
    times_ms: List[float] = []
    peak_rss = process.memory_info().rss

    processed = 0
    for images, _ in loader:
        images = images.to(device)
        t0 = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        peak_rss = max(peak_rss, process.memory_info().rss)

        processed += images.size(0)
        if processed >= max_images:
            break

    mean_time = float(mean(times_ms)) if times_ms else 0.0
    peak_ram_mb = peak_rss / (1024.0 * 1024.0)
    return mean_time, peak_ram_mb


def save_metrics(
    csv_path: Path,
    accuracy: float,
    f1_weighted: float,
    model_size_mb: float,
    inference_mean_ms: float,
    peak_ram_mb: float,
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accuracy",
                "f1_weighted",
                "model_size_mb",
                "inference_mean_ms_100_images",
                "peak_ram_mb",
                "train_samples",
                "val_samples",
                "test_samples",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "accuracy": round(accuracy, 5),
                "f1_weighted": round(f1_weighted, 5),
                "model_size_mb": round(model_size_mb, 3),
                "inference_mean_ms_100_images": round(inference_mean_ms, 3),
                "peak_ram_mb": round(peak_ram_mb, 2),
                "train_samples": train_count,
                "val_samples": val_count,
                "test_samples": test_count,
            }
        )


def save_preprocessing_artifacts(
    results_dir: Path,
    class_names: List[str],
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    skipped_corrupted: int,
    slices_per_volume: int,
) -> None:
    manifest_path = results_dir / "preprocessing_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class_name", "label", "nifti_path", "slice_idx"],
        )
        writer.writeheader()

        def write_split(name: str, items: List[Sample]) -> None:
            for s in items:
                writer.writerow(
                    {
                        "split": name,
                        "class_name": class_names[s.label],
                        "label": s.label,
                        "nifti_path": str(s.path),
                        "slice_idx": s.slice_idx,
                    }
                )

        write_split("train", train_samples)
        write_split("val", val_samples)
        write_split("test", test_samples)

    def class_count(items: List[Sample]) -> dict:
        counts = {name: 0 for name in class_names}
        for s in items:
            counts[class_names[s.label]] += 1
        return counts

    total = len(train_samples) + len(val_samples) + len(test_samples)
    distribution = {
        "slices_per_volume": slices_per_volume,
        "skipped_corrupted_files": skipped_corrupted,
        "total_samples": total,
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "split_ratios": {
            "train": round(len(train_samples) / total, 4) if total else 0.0,
            "val": round(len(val_samples) / total, 4) if total else 0.0,
            "test": round(len(test_samples) / total, 4) if total else 0.0,
        },
        "class_distribution": {
            "train": class_count(train_samples),
            "val": class_count(val_samples),
            "test": class_count(test_samples),
        },
    }

    with (results_dir / "dataset_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(distribution, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 baseline training for NIfTI MRI dataset")
    parser.add_argument("--data-root", type=str, default="data/archive", help="Dataset root containing class folders")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-inference-images", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0, help="Use a stratified subset of volumes for quick runs")
    parser.add_argument("--slices-per-volume", type=int, default=8)
    parser.add_argument("--min-total-samples", type=int, default=10000)
    parser.add_argument("--no-augment-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_STATE)

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / args.data_root
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_root}")

    volumes, class_names, skipped_corrupted = discover_volumes(data_root)

    if 0 < args.max_samples < len(volumes):
        labels = [v[1] for v in volumes]
        all_idx = list(range(len(volumes)))
        subset_idx, _ = train_test_split(
            all_idx,
            train_size=args.max_samples,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        volumes = [volumes[i] for i in subset_idx]

    train_volumes, val_volumes, test_volumes = split_volumes(volumes)
    train_samples = build_slice_samples(train_volumes, max(1, args.slices_per_volume))
    val_samples = build_slice_samples(val_volumes, max(1, args.slices_per_volume))
    test_samples = build_slice_samples(test_volumes, max(1, args.slices_per_volume))

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    if total_samples < args.min_total_samples:
        raise ValueError(
            f"Generated samples ({total_samples}) are below required minimum ({args.min_total_samples}). "
            f"Increase --slices-per-volume."
        )

    save_preprocessing_artifacts(
        results_dir=results_dir,
        class_names=class_names,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        skipped_corrupted=skipped_corrupted,
        slices_per_volume=args.slices_per_volume,
    )

    train_ds = NiftiSliceDataset(train_samples, augment=not args.no_augment_train)
    val_ds = NiftiSliceDataset(val_samples, augment=False)
    test_ds = NiftiSliceDataset(test_samples, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_val, y_val_pred = evaluate(model, val_loader, device)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = models_dir / "baseline_mobilenet_v2.pt"
    torch.save(model.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024.0 * 1024.0)

    y_test, y_test_pred = evaluate(model, test_loader, device)
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))

    report = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with (results_dir / "baseline_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    inference_mean_ms, peak_ram_mb = measure_inference(
        model,
        test_loader,
        device,
        max_images=args.max_inference_images,
    )

    save_metrics(
        csv_path=results_dir / "baseline_metrics.csv",
        accuracy=test_acc,
        f1_weighted=test_f1,
        model_size_mb=model_size_mb,
        inference_mean_ms=inference_mean_ms,
        peak_ram_mb=peak_ram_mb,
        train_count=len(train_ds),
        val_count=len(val_ds),
        test_count=len(test_ds),
    )

    summary = {
        "dataset_root": str(data_root),
        "class_names": class_names,
        "slices_per_volume": args.slices_per_volume,
        "train_volumes": len(train_volumes),
        "val_volumes": len(val_volumes),
        "test_volumes": len(test_volumes),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "total_samples": len(train_ds) + len(val_ds) + len(test_ds),
        "device": str(device),
        "accuracy": round(test_acc, 5),
        "f1_weighted": round(test_f1, 5),
        "model_size_mb": round(model_size_mb, 3),
        "inference_mean_ms_100_images": round(inference_mean_ms, 3),
        "peak_ram_mb": round(peak_ram_mb, 2),
    }
    with (results_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 1 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
