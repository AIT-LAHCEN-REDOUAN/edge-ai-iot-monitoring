import os
from datetime import datetime, timezone
from src.thingsboard.mqtt_client import TBPublisher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def parse_summary(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                data[k] = float(v) if "." in v or v.isdigit() else v
            except Exception:
                data[k] = v
    return data

def main():
    summ_path = os.path.join(ROOT, "results", "collective", "summary.txt")
    if not os.path.exists(summ_path):
        raise FileNotFoundError(summ_path)
    host = os.environ.get("TB_HOST", "localhost")
    port = int(os.environ.get("TB_PORT", "1883"))
    token = os.environ.get("TB_TOKEN", "collective").strip()
    if not token:
        raise RuntimeError("TB_TOKEN required")
    client = TBPublisher(host=host, port=port, token=token, client_id="collective-summary")
    client.connect()
    stats = parse_summary(summ_path)
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collective_accuracy": float(stats.get("collective_accuracy", 0.0)),
        "best_individual_accuracy": float(stats.get("best_individual_accuracy", 0.0)),
        "consensus_rate": float(stats.get("consensus_rate", 0.0)),
        "revalidation_improvements": float(stats.get("revalidation_improvements", 0.0))
    }
    client.publish_telemetry(payload)
    client.disconnect()

if __name__ == "__main__":
    main()
