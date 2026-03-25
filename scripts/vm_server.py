import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import psutil
from torchvision import models as tv_models
from datetime import datetime, timezone
from src.thingsboard.mqtt_client import TBPublisher

def build_model():
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def load_weights(model, vm_name, override=None):
    profiles_path = "/app/results/deployment/vm_profiles.json"
    technique = override if override else "baseline"
    if not override and os.path.exists(profiles_path):
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        if vm_name in profiles and "selected_technique" in profiles[vm_name]:
            technique = profiles[vm_name]["selected_technique"]
    paths = {
        "baseline": "/app/models/baseline/best_model.pth",
        "q1": "/app/models/optimized/q1_dynamic_quantized_model.pth",
        "q2": "/app/models/optimized/q2_static_ptq_fx_model.pth",
        "q3": "/app/models/optimized/q3_qat_model.pth",
        "q4": "/app/models/optimized/q4_weight_only_fp16_model.pth",
        "q5": "/app/models/optimized/q5_mixed_precision_fp16_checkpoint.pth",
        "p1": "/app/models/optimized/p1_unstructured_pruned_model.pth",
        "p2": "/app/models/optimized/p2_structured_pruned_model.pth",
        "p3": "/app/models/optimized/p3_magnitude_pruned_model.pth",
    }
    key = technique.lower()
    if key not in paths:
        key = "baseline"
    weight_path = paths[key]
    loaded_technique = technique
    # Attempt robust load; if many keys don't match, fallback to baseline
    def try_load(path, tag):
        ms = model.state_dict()
        try:
            state = torch.load(path, map_location="cpu")
            # Accept multiple formats: Module, checkpoint dict, plain state_dict
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Optionally strip common prefixes like "module."
            def normalize_keys(sd):
                out = {}
                for k, v in sd.items():
                    kk = k
                    if kk.startswith("module."):
                        kk = kk[len("module."):]
                    if kk.startswith("model."):
                        kk = kk[len("model."):]
                    out[kk] = v
                return out
            state = normalize_keys(state)
            matched = {}
            for k, v in state.items():
                if k in ms:
                    if v.dtype != ms[k].dtype:
                        v = v.to(ms[k].dtype)
                    matched[k] = v
            # Compute ratio relative to provided checkpoint keys to be robust
            match_ratio = len(matched) / max(1, len(state))
            ms.update(matched)
            model.load_state_dict(ms, strict=False)
            return True, tag, match_ratio
        except Exception:
            return False, "baseline", 0.0

    ok, tag, ratio = try_load(weight_path, technique)
    if not ok or ratio < 0.5:
        # Fallback to baseline weights for consistency
        base_path = paths["baseline"]
        try_load(base_path, "baseline")
        loaded_technique = "baseline"
    else:
        loaded_technique = tag
    return model, loaded_technique

def npy_to_tensor(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3 and arr.shape[0] != 1:
        arr = arr[None, ...]
    t = torch.from_numpy(arr).float()
    t = t.unsqueeze(0)
    return t

app = Flask(__name__)

vm_name = os.environ.get("VM_NAME", "vm1")
model = build_model()
model, technique_loaded = load_weights(model, vm_name)
model.eval()

TB_ENABLE = os.environ.get("TB_ENABLE", "false").lower() in ("1", "true", "yes", "on")
TB_HOST = os.environ.get("TB_HOST", "thingsboard")
TB_PORT = int(os.environ.get("TB_PORT", "1883"))
TB_TOKEN = os.environ.get("TB_TOKEN", "").strip()
tb_publisher = None
if TB_ENABLE and TB_TOKEN:
    tb_publisher = TBPublisher(host=TB_HOST, port=TB_PORT, token=TB_TOKEN, client_id=f"{vm_name}-client")
    tb_publisher.connect()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "vm": vm_name, "technique": technique_loaded})

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    slice_path = data.get("slice_path")
    req_technique = data.get("technique", None)
    global model, technique_loaded
    if req_technique and req_technique != technique_loaded:
        model, technique_loaded = load_weights(model, vm_name, override=req_technique)
    patient_id = data.get("patient_id", None)
    if not slice_path or not os.path.exists(slice_path):
        return jsonify({"error": "missing_or_invalid_path"}), 400
    x = npy_to_tensor(slice_path)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
    infer_ms = (time.perf_counter() - start) * 1000.0
    proc = psutil.Process(os.getpid())
    cpu_pct = psutil.cpu_percent(interval=None)
    mem_info = proc.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    ram_pct = psutil.virtual_memory().percent
    result = {
        "vm_id": vm_name,
        "technique": technique_loaded,
        "prediction": pred,
        "confidence": conf,
        "inference_time_ms": infer_ms,
        "cpu_usage_pct": cpu_pct,
        "ram_usage_mb": ram_mb,
        "ram_usage_pct": ram_pct
    }
    # ThingsBoard telemetry (optional)
    if tb_publisher is not None:
        payload = {
            "vm_id": vm_name.upper(),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "technique": technique_loaded,
            "prediction": pred,
            "confidence": conf,
            "inference_time_ms": round(infer_ms, 3),
            "cpu_usage_pct": round(cpu_pct, 1),
            "ram_usage_mb": round(ram_mb, 1),
            "patient_id": patient_id if patient_id is not None else ""
        }
        try:
            tb_publisher.publish_telemetry(payload)
        except Exception:
            pass
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
