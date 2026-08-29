"""Dev-only mock sender for the Dynamic Hybrid Mesh feature (mesh.enabled).

Simulates UDP broadcast traffic from Node A (ZTE phone) and Node B (ESP32-S3
scanner), matching the exact wire format `tarno_backend/integrations/mesh/payload.py`
expects, so Phase 1 (this PC's mesh plugin) can be exercised end-to-end
without any real phone/ESP32 hardware.

Usage:
    py -3.12 workspace/debug/tools/mesh_mock_sender.py --scenario full_mesh
    py -3.12 workspace/debug/tools/mesh_mock_sender.py --scenario pc_fallback
    py -3.12 workspace/debug/tools/mesh_mock_sender.py --scenario solo_pc

Not imported by the app - a standalone verification tool only.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

DEFAULT_PORT = 47800


def send(sock: socket.socket, port: int, sender_node: str, event_type: str, payload: dict) -> None:
    message = {
        "sender_node": sender_node,
        "target_hub": "DYNAMIC_BROADCAST",
        "timestamp": time.time(),
        "event_type": event_type,
        "payload": payload,
    }
    sock.sendto(json.dumps(message).encode("utf-8"), ("255.255.255.255", port))
    print(f"-> {sender_node} {event_type}: {payload}")


def run(scenario: str, port: int, duration_seconds: float) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    start = time.time()
    print(f"Mesh-Mock-Sender: Szenario '{scenario}' auf Port {port} für {duration_seconds:.0f}s")
    print("(Ctrl+C zum vorzeitigen Beenden)\n")

    try:
        while time.time() - start < duration_seconds:
            if scenario == "full_mesh":
                send(sock, port, "WIKO_VIEW5PLUS", "HEARTBEAT", {})
                send(sock, port, "ZTE_BLADE_V70", "USER_STATE_CHANGE", {"screen_state": "ON", "battery_level": 78})
            elif scenario == "pc_fallback":
                # No WIKO heartbeat at all - simulates Wiko being offline/unreachable.
                send(sock, port, "ZTE_BLADE_V70", "USER_STATE_CHANGE", {"screen_state": "ON", "battery_level": 62})
                send(sock, port, "ESP32_S3_SCANNER", "PROXIMITY_UPDATE", {"rssi_ble": -55, "rssi_wifi": -48})
            elif scenario == "solo_pc":
                # Only ESP32 - no phones present at all.
                send(sock, port, "ESP32_S3_SCANNER", "PROXIMITY_UPDATE", {"rssi_ble": -70, "rssi_wifi": -65})
            else:
                raise SystemExit(f"Unbekanntes Szenario: {scenario}")

            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("\nMesh-Mock-Sender beendet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["full_mesh", "pc_fallback", "solo_pc"], default="pc_fallback")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=30.0, help="Sekunden")
    args = parser.parse_args()
    run(args.scenario, args.port, args.duration)
