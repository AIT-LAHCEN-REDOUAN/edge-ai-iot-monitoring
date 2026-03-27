import os
import json
import pandas as pd
from datetime import datetime, timezone
from src.thingsboard.mqtt_client import TBPublisher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    csv_path = os.path.join(ROOT, "results", "deployment", "vm_tech_matrix.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    host = os.environ.get("TB_HOST", "localhost")
    port = int(os.environ.get("TB_PORT", "1883"))
    token = os.environ.get("TB_TOKEN", "collective").strip()
    if not token:
        raise RuntimeError("TB_TOKEN required")
    client = TBPublisher(host=host, port=port, token=token, client_id="scores-publisher")
    client.connect()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"timestamp": ts}
    for _, r in df.iterrows():
        vm = str(r["vm"])
        tech_req = str(r["technique_requested"]).lower()
        lat = float(r.get("latency_ms_mean", 0.0))
        acc = float(r.get("accuracy", 0.0))
        cpu = float(r.get("cpu_pct_mean", 0.0))
        ram = float(r.get("ram_mb_mean", 0.0))
        payload[f"score_{vm}_{tech_req}_lat"] = round(lat, 3)
        payload[f"score_{vm}_{tech_req}_acc"] = round(acc, 4)
        payload[f"score_{vm}_{tech_req}_cpu"] = round(cpu, 1)
        payload[f"score_{vm}_{tech_req}_ram"] = round(ram, 1)
    client.publish_telemetry(payload)
    client.disconnect()

if __name__ == "__main__":
    main()
