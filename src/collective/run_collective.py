import os
import json
import time
import csv
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timezone
from src.thingsboard.mqtt_client import TBPublisher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def to_container_path(host_path):
    # Map absolute Windows host path inside the repo to container path under /app
    try:
        rel = os.path.relpath(host_path, ROOT)
    except Exception:
        rel = os.path.basename(host_path)
    rel = rel.replace("\\", "/")
    return "/app/" + rel

def load_vm_profiles():
    path = os.path.join(ROOT, "results", "deployment", "vm_profiles.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_endpoints():
    path = os.path.join(ROOT, "config", "vm_endpoints.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tech_accuracy():
    comp = os.path.join(ROOT, "results", "deployment", "optimization_comparison_table.csv")
    if not os.path.exists(comp):
        return {}
    df = pd.read_csv(comp)
    acc = {}
    for _, r in df.iterrows():
        tech = str(r.get("technique", "")).strip().lower()
        a = float(r.get("accuracy", 0.0))
        acc[tech] = a
    return acc

def sample_test_slices(n=10):
    meta = os.path.join(ROOT, "data", "metadata", "processed_slices_metadata.csv")
    df = pd.read_csv(meta)
    df = df[(df["split"] == "test") & (df["status"] == "success")]
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df

def request_infer(url, slice_path, patient_id=None):
    t0 = time.perf_counter()
    payload = {"slice_path": slice_path}
    if patient_id is not None:
        payload["patient_id"] = str(patient_id)
    r = requests.post(url + "/infer", json=payload, timeout=60)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        return None
    data = r.json()
    data["network_rtt_ms"] = dt
    return data

def weighted_vote(responses, tech_acc):
    scores = {0: 0.0, 1: 0.0}
    weights_used = []
    for vm_id, resp in responses.items():
        if resp is None:
            continue
        tech = str(resp.get("technique", "baseline")).lower()
        hist_acc = tech_acc.get(tech, 0.9)
        conf = float(resp.get("confidence", 0.0))
        pred = int(resp.get("prediction", 0))
        w = hist_acc * conf
        scores[pred] += w
        weights_used.append(w)
    if scores[0] == 0 and scores[1] == 0:
        return 0, 0.0
    pred = 0 if scores[0] >= scores[1] else 1
    total = scores[0] + scores[1]
    conf = scores[pred] / total
    return pred, conf

def neighbors(df, row, k=2):
    subject = row["subject_id"]
    idx = int(row["slice_index"])
    neigh = df[(df["subject_id"] == subject) & (df["slice_index"].between(idx - k, idx + k))]
    return neigh

def main():
    profiles = load_vm_profiles()
    endpoints = load_endpoints()
    tech_acc = load_tech_accuracy()
    df = sample_test_slices(10)
    tb_enable = os.environ.get("TB_COLLECTIVE_ENABLE", "false").lower() in ("1", "true", "yes", "on")
    tb_token = os.environ.get("TB_COLLECTIVE_TOKEN", "").strip()
    tb_host = os.environ.get("TB_COLLECTIVE_HOST", "localhost")
    tb_port = int(os.environ.get("TB_COLLECTIVE_PORT", "1883"))
    tb_pub = None
    if tb_enable and tb_token:
        tb_pub = TBPublisher(host=tb_host, port=tb_port, token=tb_token, client_id="collective-client")
        tb_pub.connect()
    results_dir = os.path.join(ROOT, "results", "collective")
    os.makedirs(results_dir, exist_ok=True)
    detail_path = os.path.join(results_dir, "detail.csv")
    summary_path = os.path.join(results_dir, "summary.txt")
    total = 0
    correct_individual_best = 0
    correct_collective = 0
    consensus_count = 0
    reval_gain = 0
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Collective eval"):
        slice_host = r["slice_filepath"]
        slice_cont = to_container_path(slice_host)
        y_true = int(r["label"])
        responses = {}
        overloaded = set()
        patient_id = str(r.get("subject_id", ""))
        for vm in ["vm1", "vm2", "vm3"]:
            url = endpoints.get(vm)
            if not url:
                continue
            resp = request_infer(url, slice_cont, patient_id=patient_id)
            if resp is None:
                responses[vm] = None
                continue
            if resp.get("cpu_usage_pct", 0) > 85 or resp.get("ram_usage_pct", 0) > 90:
                overloaded.add(vm)
                responses[vm] = None
                continue
            responses[vm] = resp
        preds = {vm: (v["prediction"] if v else None) for vm, v in responses.items()}
        confs = {vm: (v["confidence"] if v else None) for vm, v in responses.items()}
        if all(v is not None for v in preds.values()):
            consensus = int(len(set(preds.values())) == 1)
        else:
            consensus = 0
        pred_collective, conf_collective = weighted_vote(responses, tech_acc)
        if conf_collective < 0.7:
            neigh_df = neighbors(df, r, k=2)
            agg_scores = {0: 0.0, 1: 0.0}
            base_scores = {0: 0.0, 1: 0.0}
            for vm, resp in responses.items():
                if resp is None:
                    continue
                tech = str(resp.get("technique", "baseline")).lower()
                hist_acc = tech_acc.get(tech, 0.9)
                conf = float(resp.get("confidence", 0.0))
                pred = int(resp.get("prediction", 0))
                w = hist_acc * conf
                base_scores[pred] += w
            for _, nr in neigh_df.iterrows():
                if nr["slice_filepath"] == slice_host:
                    continue
                s2 = to_container_path(nr["slice_filepath"])
                for vm in ["vm1", "vm2", "vm3"]:
                    if vm in overloaded:
                        continue
                    url = endpoints.get(vm)
                    if not url:
                        continue
                    r2 = request_infer(url, s2, patient_id=patient_id)
                    if r2 is None:
                        continue
                    tech = str(r2.get("technique", "baseline")).lower()
                    hist_acc = tech_acc.get(tech, 0.9)
                    conf2 = float(r2.get("confidence", 0.0))
                    pred2 = int(r2.get("prediction", 0))
                    w2 = hist_acc * conf2
                    agg_scores[pred2] += w2
            tot0 = base_scores[0] + agg_scores[0]
            tot1 = base_scores[1] + agg_scores[1]
            if tot0 == 0 and tot1 == 0:
                pass
            else:
                pred2 = 0 if tot0 >= tot1 else 1
                conf2 = (tot0 if pred2 == 0 else tot1) / (tot0 + tot1)
                if conf2 > conf_collective:
                    pred_collective = pred2
                    conf_collective = conf2
                    reval_gain += 1
        best_individual = None
        best_conf = -1.0
        for vm, resp in responses.items():
            if resp is None:
                continue
            c = float(resp.get("confidence", 0.0))
            if c > best_conf:
                best_conf = c
                best_individual = int(resp.get("prediction", 0))
        total += 1
        if best_individual is not None and best_individual == y_true:
            correct_individual_best += 1
        if pred_collective == y_true:
            correct_collective += 1
        consensus_count += consensus
        if tb_pub is not None:
            # average of VM inference times used in this decision
            used_times = [resp.get("inference_time_ms", 0.0) for resp in responses.values() if resp is not None]
            avg_infer_ms = float(np.mean(used_times)) if used_times else 0.0
            payload = {
                "vm_id": "COLLECTIVE",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "collective_pred": int(pred_collective),
                "collective_confidence": float(conf_collective),
                "consensus": int(consensus),
                "overloaded_count": len(overloaded),
                "avg_inference_time_ms": round(avg_infer_ms, 3),
                "patient_id": str(r.get("subject_id", ""))
            }
            try:
                tb_pub.publish_telemetry(payload)
            except Exception:
                pass
        rows.append({
            "slice": slice_host,
            "y_true": y_true,
            "pred_vm1": preds.get("vm1"),
            "conf_vm1": confs.get("vm1"),
            "pred_vm2": preds.get("vm2"),
            "conf_vm2": confs.get("vm2"),
            "pred_vm3": preds.get("vm3"),
            "conf_vm3": confs.get("vm3"),
            "collective_pred": pred_collective,
            "collective_conf": conf_collective,
            "consensus": consensus,
            "overloaded": ",".join(sorted(list(overloaded)))
        })
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    acc_collective = correct_collective / total if total else 0.0
    acc_individual_best = correct_individual_best / total if total else 0.0
    consensus_rate = consensus_count / total if total else 0.0
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total={total}\n")
        f.write(f"collective_accuracy={acc_collective:.4f}\n")
        f.write(f"best_individual_accuracy={acc_individual_best:.4f}\n")
        f.write(f"consensus_rate={consensus_rate:.4f}\n")
        f.write(f"revalidation_improvements={reval_gain}\n")

if __name__ == "__main__":
    main()
