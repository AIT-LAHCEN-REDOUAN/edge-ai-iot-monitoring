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
        
        # Set up callbacks to track connection state
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connection is established"""
        if rc == 0:
            self._connected = True
            print(f"[TBPublisher] Connected to ThingsBoard at {self.host}:{self.port}")
        else:
            self._connected = False
            print(f"[TBPublisher] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected"""
        self._connected = False
        if rc != 0:
            print(f"[TBPublisher] Unexpected disconnect (code {rc})")

    def connect(self, timeout_s: float = 4.0, max_retries: int = 5) -> bool:
        """Connect to ThingsBoard with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"[TBPublisher] Connection attempt {attempt + 1}/{max_retries}...")
                self._client.connect(self.host, self.port, keepalive=self.keepalive)
                self._client.loop_start()
                
                # Wait for connection to establish
                t0 = time.time()
                while not self._connected and time.time() - t0 < timeout_s:
                    time.sleep(0.1)
                
                if self._connected:
                    print(f"[TBPublisher] Successfully connected as {self.client_id}")
                    return True
                else:
                    print(f"[TBPublisher] Connection timeout after {timeout_s}s")
                    self._client.loop_stop()
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                        print(f"[TBPublisher] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        
            except Exception as e:
                print(f"[TBPublisher] Connection error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[TBPublisher] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        print(f"[TBPublisher] Failed to connect after {max_retries} attempts")
        return False

    def publish_telemetry(self, payload: Dict[str, Any]) -> bool:
        topic = "v1/devices/me/telemetry"
        try:
            if not self._connected:
                print(f"[TBPublisher] Not connected. Cannot publish.")
                return False
                
            data = json.dumps(payload)
            result = self._client.publish(topic, data, qos=1)
            result.wait_for_publish(timeout=3.0)
            return result.is_published()
        except Exception as e:
            print(f"[TBPublisher] Publish error: {e}")
            return False

    def disconnect(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
        except Exception:
            pass