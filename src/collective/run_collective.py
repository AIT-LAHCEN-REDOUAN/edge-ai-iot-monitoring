import os
import sys

# Make project root importable so `from src....` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import time
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.thingsboard.mqtt_client import TBPublisher


def to_container_path(host_path: str) -> str:
    """
    Map an absolute Windows/Linux host path inside the repo
    to the Docker container path under /app.
    """
    try:
        rel = os.path.relpath(host_path, ROOT)
    except Exception:
        rel = os.path.basename(host_path)

    rel = rel.replace("\\", "/")
    return "/app/" + rel


def load_vm_profiles() -> dict:
    path = os.path.join(ROOT, "results", "deployment", "vm_profiles.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_endpoints() -> dict:
    path = os.path.join(ROOT, "config", "vm_endpoints.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing endpoints file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tech_accuracy() -> dict:
    comp = os.path.join(ROOT, "results", "deployment", "optimization_comparison_table.csv")
    if not os.path.exists(comp):
        return {}

    df = pd.read_csv(comp)
    acc = {}

    for _, row in df.iterrows():
        tech = str(row.get("technique", "")).strip().lower()
        if not tech:
            continue
        try:
            acc[tech] = float(row.get("accuracy", 0.0))
        except Exception:
            acc[tech] = 0.0

    return acc


def sample_test_slices(n: int = 10) -> pd.DataFrame:
    meta = os.path.join(ROOT, "data", "metadata", "processed_slices_metadata.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"Missing metadata file: {meta}")

    df = pd.read_csv(meta)
    df = df[(df["split"] == "test") & (df["status"] == "success")].copy()

    if df.empty:
        raise ValueError("No successful test slices found.")

    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    return df.reset_index(drop=True)


def request_infer(url: str, slice_path: str, patient_id=None):
    """
    Call a VM inference endpoint and measure network round trip time.
    """
    t0 = time.perf_counter()
    payload = {"slice_path": slice_path}

    if patient_id is not None:
        payload["patient_id"] = str(patient_id)

    try:
        response = requests.post(f"{url}/infer", json=payload, timeout=60)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code != 200:
            return None

        data = response.json()
        data["network_rtt_ms"] = dt_ms
        return data

    except Exception:
        return None


def weighted_vote(responses: dict, tech_acc: dict):
    """
    Weighted vote = historical accuracy * confidence.
    """
    scores = {0: 0.0, 1: 0.0}

    for vm_id, resp in responses.items():
        if resp is None:
            continue

        tech = str(resp.get("technique", "baseline")).lower()
        hist_acc = tech_acc.get(tech, 0.9)
        conf = float(resp.get("confidence", 0.0))
        pred = int(resp.get("prediction", 0))

        weight = hist_acc * conf
        scores[pred] += weight

    if scores[0] == 0 and scores[1] == 0:
        return 0, 0.0

    pred = 0 if scores[0] >= scores[1] else 1
    total = scores[0] + scores[1]
    conf = scores[pred] / total if total > 0 else 0.0

    return pred, conf


def neighbors(df: pd.DataFrame, row: pd.Series, k: int = 2) -> pd.DataFrame:
    subject = row["subject_id"]
    idx = int(row["slice_index"])

    neigh = df[
        (df["subject_id"] == subject)
        & (df["slice_index"].between(idx - k, idx + k))
    ].copy()

    return neigh


def build_tb_publisher():
    tb_enable = os.environ.get("TB_COLLECTIVE_ENABLE", "false").lower() in ("1", "true", "yes", "on")
    tb_token = os.environ.get("TB_COLLECTIVE_TOKEN", "").strip()
    tb_host = os.environ.get("TB_COLLECTIVE_HOST", "localhost")
    tb_port = int(os.environ.get("TB_COLLECTIVE_PORT", "1883"))

    if not tb_enable or not tb_token:
        return None

    try:
        publisher = TBPublisher(
            host=tb_host,
            port=tb_port,
            token=tb_token,
            client_id="collective-client"
        )
        publisher.connect()
        return publisher
    except Exception as e:
        print(f"[WARNING] ThingsBoard connection failed: {e}")
        return None


def main():
    print("[INFO] Starting collective evaluation...")

    profiles = load_vm_profiles()
    endpoints = load_endpoints()
    tech_acc = load_tech_accuracy()
    df = sample_test_slices(10)

    tb_pub = build_tb_publisher()

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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collective eval"):
        slice_host = row["slice_filepath"]
        slice_cont = to_container_path(slice_host)
        y_true = int(row["label"])
        patient_id = str(row.get("subject_id", ""))

        responses = {}
        overloaded = set()

        for vm in ["vm1", "vm2", "vm3"]:
            url = endpoints.get(vm)
            if not url:
                responses[vm] = None
                continue

            resp = request_infer(url, slice_cont, patient_id=patient_id)

            if resp is None:
                responses[vm] = None
                continue

            cpu_usage = float(resp.get("cpu_usage_pct", 0))
            ram_usage = float(resp.get("ram_usage_pct", 0))

            if cpu_usage > 85 or ram_usage > 90:
                overloaded.add(vm)
                responses[vm] = None
                continue

            responses[vm] = resp

        preds = {vm: (resp["prediction"] if resp else None) for vm, resp in responses.items()}
        confs = {vm: (resp["confidence"] if resp else None) for vm, resp in responses.items()}

        available_preds = [p for p in preds.values() if p is not None]
        consensus = int(len(available_preds) == 3 and len(set(available_preds)) == 1)

        pred_collective, conf_collective = weighted_vote(responses, tech_acc)

        if conf_collective < 0.7:
            neigh_df = neighbors(df, row, k=2)

            agg_scores = {0: 0.0, 1: 0.0}
            base_scores = {0: 0.0, 1: 0.0}

            for vm, resp in responses.items():
                if resp is None:
                    continue

                tech = str(resp.get("technique", "baseline")).lower()
                hist_acc = tech_acc.get(tech, 0.9)
                conf = float(resp.get("confidence", 0.0))
                pred = int(resp.get("prediction", 0))
                weight = hist_acc * conf
                base_scores[pred] += weight

            for _, neigh_row in neigh_df.iterrows():
                if neigh_row["slice_filepath"] == slice_host:
                    continue

                s2 = to_container_path(neigh_row["slice_filepath"])

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
                    weight2 = hist_acc * conf2
                    agg_scores[pred2] += weight2

            tot0 = base_scores[0] + agg_scores[0]
            tot1 = base_scores[1] + agg_scores[1]

            if not (tot0 == 0 and tot1 == 0):
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
            used_times = [
                float(resp.get("inference_time_ms", 0.0))
                for resp in responses.values()
                if resp is not None
            ]
            avg_infer_ms = float(np.mean(used_times)) if used_times else 0.0

            payload = {
                "vm_id": "COLLECTIVE",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "collective_pred": int(pred_collective),
                "collective_confidence": float(conf_collective),
                "consensus": int(consensus),
                "overloaded_count": len(overloaded),
                "avg_inference_time_ms": round(avg_infer_ms, 3),
                "patient_id": patient_id
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

    if rows:
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    acc_collective = correct_collective / total if total else 0.0
    acc_individual_best = correct_individual_best / total if total else 0.0
    consensus_rate = consensus_count / total if total else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total={total}\n")
        f.write(f"collective_accuracy={acc_collective:.4f}\n")
        f.write(f"best_individual_accuracy={acc_individual_best:.4f}\n")
        f.write(f"consensus_rate={consensus_rate:.4f}\n")
        f.write(f"revalidation_improvements={reval_gain}\n")

    print("[INFO] Collective evaluation completed.")
    print(f"[INFO] detail.csv saved to: {detail_path}")
    print(f"[INFO] summary.txt saved to: {summary_path}")


if __name__ == "__main__":
    main()