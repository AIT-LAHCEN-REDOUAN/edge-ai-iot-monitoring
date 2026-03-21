import argparse
import copy
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Missing dependency 'nibabel'. Install it with: pip install nibabel") from exc


TECHNIQUES = ["P1", "P2", "P3"]


@dataclass
class ManifestSample:
    path: Path
    label: int
    slice_idx: int


class ManifestNiftiDataset(Dataset):
    def __init__(self, samples: List[ManifestSample], image_size: int = 224) -> None:
        self.samples = samples
        self.image_size = image_size
        self._cached_path: str | None = None
        self._cached_proxy = None

    def __len__(self) -> int:
        return len(self.samples)

    def _get_proxy(self, path: Path):
        path_str = str(path)
        if self._cached_path != path_str or self._cached_proxy is None:
            nii = nib.load(path_str)
            self._cached_path = path_str
            self._cached_proxy = nii.dataobj
        return self._cached_proxy

    @staticmethod
    def _read_slice(proxy, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), proxy.shape[2] - 1))
        try:
            return np.asanyarray(proxy[:, :, idx], dtype=np.float32)
        except Exception:
            fallback = proxy.shape[2] // 2
            return np.asanyarray(proxy[:, :, fallback], dtype=np.float32)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        min_v = float(x.min())
        max_v = float(x.max())
        if max_v > min_v:
            return (x - min_v) / (max_v - min_v)
        return np.zeros_like(x, dtype=np.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        proxy = self._get_proxy(sample.path)
        channels = [
            self._normalize(self._read_slice(proxy, sample.slice_idx - 1)),
            self._normalize(self._read_slice(proxy, sample.slice_idx)),
            self._normalize(self._read_slice(proxy, sample.slice_idx + 1)),
        ]
        stacked = np.stack(channels, axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        return tensor, sample.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 pruning P1-P3")
    parser.add_argument("--manifest", type=str, default="results/preprocessing_manifest.csv")
    parser.add_argument("--baseline-weights", type=str, default="models/baseline_mobilenet_v2.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=400)
    parser.add_argument("--p1-amount", type=float, default=0.50)
    parser.add_argument("--p2-amount", type=float, default=0.30)
    parser.add_argument("--p3-iter-steps", type=int, default=3)
    parser.add_argument("--p3-step-amount", type=float, default=0.20)
    return parser.parse_args()


def load_manifest_samples(manifest_path: Path, split: str, max_samples: int = 0) -> List[ManifestSample]:
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].copy()
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    out: List[ManifestSample] = []
    for row in df.itertuples(index=False):
        out.append(ManifestSample(path=Path(row.nifti_path), label=int(row.label), slice_idx=int(row.slice_idx)))
    return out


def make_base_model(num_classes: int, weights_path: Path) -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def model_size_mb(model: nn.Module) -> float:
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), temp_path)
        return Path(temp_path).stat().st_size / (1024.0 * 1024.0)
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def sparsity(model: nn.Module) -> float:
    total = 0
    zero = 0
    for p in model.parameters():
        arr = p.detach().cpu()
        total += arr.numel()
        zero += int((arr == 0).sum().item())
    if total == 0:
        return 0.0
    return zero / total


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    all_true: List[int] = []
    all_pred: List[int] = []
    timings: List[float] = []

    for images, labels in loader:
        t0 = time.perf_counter()
        logits = model(images)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)
        peak_rss = max(peak_rss, process.memory_info().rss)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(labels.numpy().tolist())

    acc = float(accuracy_score(all_true, all_pred)) if all_true else 0.0
    f1w = float(f1_score(all_true, all_pred, average="weighted")) if all_true else 0.0
    return {
        "accuracy": round(acc, 5),
        "f1_weighted": round(f1w, 5),
        "inference_mean_ms": round(float(mean(timings)) if timings else 0.0, 3),
        "peak_ram_mb": round(peak_rss / (1024.0 * 1024.0), 2),
    }


def apply_p1_unstructured(model: nn.Module, amount: float) -> nn.Module:
    m = copy.deepcopy(model)
    to_prune = []
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            to_prune.append((module, "weight"))
    prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in to_prune:
        prune.remove(module, "weight")
    return m.eval()


def apply_p2_structured(model: nn.Module, amount: float) -> nn.Module:
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return m.eval()


def apply_p3_magnitude_iterative(model: nn.Module, iter_steps: int, step_amount: float) -> nn.Module:
    m = copy.deepcopy(model)
    to_prune = []
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            to_prune.append((module, "weight"))
    for _ in range(max(1, iter_steps)):
        prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=step_amount)
    for module, _ in to_prune:
        prune.remove(module, "weight")
    return m.eval()


def build_comparative_assets(results_dir: Path) -> None:
    q_path = results_dir / "quantization_results.csv"
    p_path = results_dir / "pruning_results.csv"
    if not q_path.exists() or not p_path.exists():
        return

    q_df = pd.read_csv(q_path)
    p_df = pd.read_csv(p_path)

    q_df["family"] = "quantization"
    p_df["family"] = "pruning"

    if "effective_model_size_mb" not in q_df.columns:
        q_df["effective_model_size_mb"] = q_df["model_size_mb"]

    full = pd.concat([q_df, p_df], ignore_index=True)
    full.to_csv(results_dir / "phase2_comparative_table.csv", index=False)

    ok_df = full[full["status"] == "ok"].copy()
    if ok_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(ok_df["effective_model_size_mb"], ok_df["accuracy"])
    for row in ok_df.itertuples(index=False):
        plt.annotate(row.technique, (row.effective_model_size_mb, row.accuracy), fontsize=8)
    plt.xlabel("Effective model size (MB)")
    plt.ylabel("Accuracy")
    plt.title("Phase 2: Taille vs Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "phase2_size_vs_precision.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(ok_df["inference_mean_ms"], ok_df["accuracy"])
    for row in ok_df.itertuples(index=False):
        plt.annotate(row.technique, (row.inference_mean_ms, row.accuracy), fontsize=8)
    plt.xlabel("Inference mean (ms)")
    plt.ylabel("Accuracy")
    plt.title("Phase 2: Vitesse vs Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "phase2_speed_vs_precision.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = project_root / args.manifest
    baseline_path = project_root / args.baseline_weights

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline weights: {baseline_path}")

    test_samples = load_manifest_samples(manifest_path, split="test", max_samples=args.max_eval_samples)
    train_samples = load_manifest_samples(manifest_path, split="train", max_samples=800)
    num_classes = len({s.label for s in (test_samples + train_samples)})

    loader = DataLoader(ManifestNiftiDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    base_model = make_base_model(num_classes=num_classes, weights_path=baseline_path)
    base_size = model_size_mb(base_model)

    rows: List[Dict[str, object]] = []
    configs = {
        "P1": lambda: apply_p1_unstructured(base_model, args.p1_amount),
        "P2": lambda: apply_p2_structured(base_model, args.p2_amount),
        "P3": lambda: apply_p3_magnitude_iterative(base_model, args.p3_iter_steps, args.p3_step_amount),
    }

    for tech in TECHNIQUES:
        row: Dict[str, object] = {"technique": tech, "status": "ok"}
        try:
            print(f"Running {tech}...")
            model = configs[tech]()
            metrics = evaluate_model(model, loader)
            sp = sparsity(model)
            stored_size = model_size_mb(model)
            effective_size = stored_size * (1.0 - sp)
            compression = (base_size / effective_size) if effective_size > 0 else 0.0
            row.update(metrics)
            row["model_size_mb"] = round(stored_size, 3)
            row["effective_model_size_mb"] = round(effective_size, 3)
            row["sparsity"] = round(sp, 5)
            row["compression_ratio"] = round(compression, 3)
            if tech == "P1":
                row["note"] = f"Unstructured global L1 pruning, amount={args.p1_amount}"
            elif tech == "P2":
                row["note"] = f"Structured channel pruning on Conv2d, amount={args.p2_amount}"
            else:
                row["note"] = f"Iterative magnitude pruning, steps={args.p3_iter_steps}, step_amount={args.p3_step_amount}"
        except Exception as exc:
            row["status"] = "failed"
            row["accuracy"] = None
            row["f1_weighted"] = None
            row["inference_mean_ms"] = None
            row["peak_ram_mb"] = None
            row["model_size_mb"] = None
            row["effective_model_size_mb"] = None
            row["sparsity"] = None
            row["compression_ratio"] = None
            row["note"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(row)

    out_csv = results_dir / "pruning_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "technique",
                "status",
                "accuracy",
                "f1_weighted",
                "model_size_mb",
                "effective_model_size_mb",
                "sparsity",
                "compression_ratio",
                "inference_mean_ms",
                "peak_ram_mb",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "pruning_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    build_comparative_assets(results_dir)
    print("\nPhase 2 pruning complete")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
