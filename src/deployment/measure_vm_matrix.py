import os
import json
import time
import statistics
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def to_container_path(host_path):
    try:
        rel = os.path.relpath(host_path, ROOT)
    except Exception:
        rel = os.path.basename(host_path)
    rel = rel.replace("\\", "/")
    return "/app/" + rel

def load_endpoints():
    path = os.path.join(ROOT, "config", "vm_endpoints.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_test_slices(n=10):
    meta = os.path.join(ROOT, "data", "metadata", "processed_slices_metadata.csv")
    df = pd.read_csv(meta)
    df = df[(df["split"] == "test") & (df["status"] == "success")]
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df

def request_infer(url, slice_path, patient_id=None, technique=None, timeout=60):
    payload = {"slice_path": slice_path}
    if patient_id is not None:
        payload["patient_id"] = str(patient_id)
    if technique is not None:
        payload["technique"] = technique
    t0 = time.perf_counter()
    r = requests.post(url + "/infer", json=payload, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        return None, dt
    data = r.json()
    data["network_rtt_ms"] = dt
    return data, dt

def main():
    endpoints = load_endpoints()
    df = sample_test_slices(10)
    techniques = ["baseline", "q1", "q2", "q3", "q4", "q5", "p1", "p2", "p3"]
    rows = []
    for vm, base_url in endpoints.items():
        for tech in techniques:
            preds = []
            truths = []
            times = []
            cpus = []
            rams = []
            used_techs = []
            errors = 0
            for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{vm}-{tech}"):
                cont_path = to_container_path(r["slice_filepath"])
                y_true = int(r["label"])
                patient_id = str(r.get("subject_id", ""))
                try:
                    resp, dt = request_infer(base_url, cont_path, patient_id=patient_id, technique=tech)
                except Exception:
                    errors += 1
                    continue
                if resp is None:
                    errors += 1
                    continue
                pred = int(resp.get("prediction", 0))
                preds.append(pred)
                truths.append(y_true)
                times.append(float(resp.get("inference_time_ms", dt)))
                cpus.append(float(resp.get("cpu_usage_pct", 0.0)))
                rams.append(float(resp.get("ram_usage_mb", 0.0)))
                used_techs.append(str(resp.get("technique", "")))
            n = len(preds)
            acc = float(np.mean(np.array(preds) == np.array(truths))) if n > 0 else 0.0
            lat_mean = float(np.mean(times)) if n > 0 else 0.0
            lat_std = float(statistics.pstdev(times)) if n > 1 else 0.0
            cpu_mean = float(np.mean(cpus)) if n > 0 else 0.0
            ram_mean = float(np.mean(rams)) if n > 0 else 0.0
            tech_mismatch = "" if not used_techs else ",".join(sorted(set(used_techs)))
            rows.append({
                "vm": vm,
                "technique_requested": tech,
                "technique_used_set": tech_mismatch,
                "n": n,
                "errors": errors,
                "accuracy": round(acc, 4),
                "latency_ms_mean": round(lat_mean, 3),
                "latency_ms_std": round(lat_std, 3),
                "cpu_pct_mean": round(cpu_mean, 1),
                "ram_mb_mean": round(ram_mean, 1)
            })
    out_dir = os.path.join(ROOT, "results", "deployment")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "vm_tech_matrix.csv")
    out_txt = os.path.join(out_dir, "vm_tech_matrix_summary.txt")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
