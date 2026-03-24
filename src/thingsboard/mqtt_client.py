import json
import time
from typing import Optional, Dict, Any

import paho.mqtt.client as mqtt


class TBPublisher:
    def __init__(self, host: str, port: int, token: str, client_id: Optional[str] = None, keepalive: int = 60):
        self.host = host
        self.port = port
        self.token = token
        self.keepalive = keepalive
        self.client_id = client_id
        self._client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        # ThingsBoard uses token as username without password
        self._client.username_pw_set(self.token)
        self._connected = False

    def connect(self, timeout_s: float = 4.0) -> bool:
        try:
            self._client.connect(self.host, self.port, keepalive=self.keepalive)
            self._client.loop_start()
            # Simple wait to allow connection to establish
            t0 = time.time()
            while not self._connected and time.time() - t0 < timeout_s:
                # paho-mqtt sets _state internally; we rely on publish result to confirm later
                time.sleep(0.05)
            self._connected = True
            return True
        except Exception:
            return False

    def publish_telemetry(self, payload: Dict[str, Any]) -> bool:
        topic = "v1/devices/me/telemetry"
        try:
            data = json.dumps(payload)
            result = self._client.publish(topic, data, qos=1)
            result.wait_for_publish(timeout=3.0)
            return result.is_published()
        except Exception:
            return False

    def disconnect(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

