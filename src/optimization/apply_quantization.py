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
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Missing dependency 'nibabel'. Install it with: pip install nibabel") from exc


TECHNIQUES = ["Q1", "Q2", "Q3", "Q4", "Q5"]


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
    parser = argparse.ArgumentParser(description="Phase 2 quantification Q1-Q5")
    parser.add_argument("--manifest", type=str, default="results/preprocessing_manifest.csv")
    parser.add_argument("--baseline-weights", type=str, default="models/baseline_mobilenet_v2.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means full test split")
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--qat-steps", type=int, default=60)
    return parser.parse_args()


def load_manifest_samples(manifest_path: Path, split: str, max_samples: int = 0) -> List[ManifestSample]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].copy()
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    samples: List[ManifestSample] = []
    for row in df.itertuples(index=False):
        samples.append(ManifestSample(path=Path(row.nifti_path), label=int(row.label), slice_idx=int(row.slice_idx)))
    return samples


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


def quantize_dequant_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        return t
    x = t.detach().cpu()
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return x.clone()
    qmax = (1 << bits) - 1
    scale = (x_max - x_min) / qmax
    q = torch.clamp(torch.round((x - x_min) / scale), 0, qmax)
    dq = q * scale + x_min
    return dq.to(dtype=t.dtype)


def technique_q1_dynamic(base_model: nn.Module, _: DataLoader, __: int) -> Tuple[nn.Module, str]:
    model = torch.quantization.quantize_dynamic(copy.deepcopy(base_model), {nn.Linear}, dtype=torch.qint8)
    return model, "Dynamic quantization on Linear layers (int8)."


def technique_q2_ptq_static(base_model: nn.Module, calib_loader: DataLoader, calibration_batches: int) -> Tuple[nn.Module, str]:
    import torch.ao.quantization as tq
    from torch.ao.quantization import quantize_fx as qfx

    model = copy.deepcopy(base_model).eval()
    qconfig_mapping = tq.QConfigMapping().set_global(tq.get_default_qconfig("fbgemm"))
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared = qfx.prepare_fx(model, qconfig_mapping, example_inputs)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calib_loader):
            _ = prepared(images)
            if batch_idx + 1 >= calibration_batches:
                break

    converted = qfx.convert_fx(prepared)
    return converted.eval(), "PTQ static via FX graph mode (int8 weights+activations)."


def technique_q3_qat(base_model: nn.Module, calib_loader: DataLoader, qat_steps: int) -> Tuple[nn.Module, str]:
    import torch.ao.quantization as tq
    from torch.ao.quantization import quantize_fx as qfx

    model = copy.deepcopy(base_model).train()
    qconfig_mapping = tq.QConfigMapping().set_global(tq.get_default_qat_qconfig("fbgemm"))
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared = qfx.prepare_qat_fx(model, qconfig_mapping, example_inputs)

    optimizer = optim.Adam(prepared.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    step = 0
    for images, labels in calib_loader:
        optimizer.zero_grad()
        logits = prepared(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        step += 1
        if step >= qat_steps:
            break

    prepared.eval()
    converted = qfx.convert_fx(prepared)
    return converted.eval(), "QAT with short fine-tuning then conversion to quantized model."


def technique_q4_weight_only(base_model: nn.Module, _: DataLoader, __: int) -> Tuple[nn.Module, str]:
    model = copy.deepcopy(base_model)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.copy_(quantize_dequant_tensor(p, bits=8))
    return model.eval(), "Weight-only simulation (int8 QDQ on weights, fp32 activations)."


def technique_q5_mixed_precision(base_model: nn.Module, _: DataLoader, __: int) -> Tuple[nn.Module, str]:
    model = copy.deepcopy(base_model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            bits = 8 if "classifier" in name or "features.17" in name else 4
            p.copy_(quantize_dequant_tensor(p, bits=bits))
    return model.eval(), "Mixed precision simulation (8-bit sensitive layers, 4-bit others)."


def run_technique(
    technique: str,
    builder: Callable[[nn.Module, DataLoader, int], Tuple[nn.Module, str]],
    base_model: nn.Module,
    calib_loader: DataLoader,
    eval_loader: DataLoader,
    control_steps: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {"technique": technique, "status": "ok"}
    try:
        model, note = builder(base_model, calib_loader, control_steps)
        metrics = evaluate_model(model, eval_loader)
        row.update(metrics)
        row["model_size_mb"] = round(model_size_mb(model), 3)
        row["note"] = note
    except Exception as exc:
        row["status"] = "failed"
        row["accuracy"] = None
        row["f1_weighted"] = None
        row["model_size_mb"] = None
        row["inference_mean_ms"] = None
        row["peak_ram_mb"] = None
        row["note"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    models_dir = project_root / "models" / "quantized"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = project_root / args.manifest
    baseline_path = project_root / args.baseline_weights

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline weights not found: {baseline_path}")

    train_samples = load_manifest_samples(manifest_path, split="train", max_samples=1200)
    test_samples = load_manifest_samples(manifest_path, split="test", max_samples=args.max_eval_samples)

    if not train_samples or not test_samples:
        raise ValueError("Manifest did not provide train/test samples. Run Phase 1 first.")

    num_classes = len({s.label for s in train_samples + test_samples})
    base_model = make_base_model(num_classes=num_classes, weights_path=baseline_path)

    calib_loader = DataLoader(
        ManifestNiftiDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        ManifestNiftiDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    builders = {
        "Q1": (technique_q1_dynamic, args.calibration_batches),
        "Q2": (technique_q2_ptq_static, args.calibration_batches),
        "Q3": (technique_q3_qat, args.qat_steps),
        "Q4": (technique_q4_weight_only, 0),
        "Q5": (technique_q5_mixed_precision, 0),
    }

    rows: List[Dict[str, object]] = []
    for tech in TECHNIQUES:
        builder, control_steps = builders[tech]
        print(f"Running {tech}...")
        row = run_technique(tech, builder, base_model, calib_loader, eval_loader, control_steps)
        rows.append(row)
        print(row)

    csv_path = results_dir / "quantization_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "technique",
                "status",
                "accuracy",
                "f1_weighted",
                "model_size_mb",
                "inference_mean_ms",
                "peak_ram_mb",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "quantization_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nPhase 2 quantification complete")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
