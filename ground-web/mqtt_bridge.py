"""实验室：把 MQTT cmd 写到 USB-TTL（代替尚未接线的 FS800）。

网页网关：  python server.py --mqtt
本桥接：    python mqtt_bridge.py --port COM5
Broker：    本机 Mosquitto 1883

Retain 关闭；ClientID 默认 boat01-bridge，勿与网关/网页重复。
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import paho.mqtt.client as mqtt
import serial

CMD_TOPIC = "boat/boat01/cmd"
ACK_TOPIC = "boat/boat01/ack"

_ser = None
_ser_lock = threading.Lock()
_client: mqtt.Client | None = None
_stop = threading.Event()


def _ack_from_cmd(payload: str) -> str:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "OK T=0 Y=0"
    if str(obj.get("mode", "")) == "stop":
        return "OK T=0 Y=0"
    t = max(-100, min(100, int(obj.get("T", 0))))
    y = max(-100, min(100, int(obj.get("Y", 0))))
    return f"OK T={t * 10} Y={y * 10}"


def _write_serial(text: str) -> None:
    data = text.encode("utf-8")
    if not data.endswith(b"\n"):
        data += b"\n"
    with _ser_lock:
        if _ser is None or not _ser.is_open:
            raise RuntimeError("串口未打开")
        _ser.write(data)
        _ser.flush()


def _on_message(_c, _u, msg) -> None:
    if msg.topic != CMD_TOPIC or _client is None:
        return
    payload = msg.payload.decode("utf-8", errors="replace").strip()
    if not payload.startswith("{") or "}" not in payload:
        return
    compact = payload[: payload.rfind("}") + 1]
    try:
        _write_serial(compact)
    except Exception as exc:
        print("写串口失败:", exc)


def _reader(mock: bool) -> None:
    buf = bytearray()
    while not _stop.is_set():
        if mock:
            time.sleep(0.2)
            continue
        chunk = b""
        with _ser_lock:
            ser = _ser
            if ser is None or not ser.is_open:
                break
            try:
                n = ser.in_waiting
                if n:
                    chunk = ser.read(n)
            except Exception:
                chunk = b""
        if not chunk:
            time.sleep(0.05)
            continue
        buf.extend(chunk)
        while True:
            n_lf = buf.find(b"\n")
            n_cr = buf.find(b"\r")
            if n_lf < 0 and n_cr < 0:
                if len(buf) > 256:
                    buf.clear()
                break
            cuts = [i for i in (n_lf, n_cr) if i >= 0]
            nl = min(cuts)
            line = buf[:nl].decode("utf-8", errors="replace").strip()
            skip = nl + 1
            if skip < len(buf) and buf[nl] == 0x0D and buf[skip] == 0x0A:
                skip += 1
            del buf[:skip]
            if line and _client is not None:
                _client.publish(ACK_TOPIC, line, qos=1, retain=False)


def main() -> None:
    global _ser, _client
    parser = argparse.ArgumentParser(description="MQTT ↔ 串口桥（实验室代替 FS800）")
    parser.add_argument("--port", default="", help="COM 口，如 COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-user", default="")
    parser.add_argument("--mqtt-password", default="")
    parser.add_argument("--mqtt-client-id", default="boat01-bridge")
    parser.add_argument("--mock", action="store_true", help="无串口，收到 cmd 就回 OK")
    args = parser.parse_args()

    if not args.mock:
        if not args.port:
            raise SystemExit("请指定 --port COMx，或用 --mock")
        _ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=0.05, write_timeout=0.3)

    kwargs = {"client_id": args.mqtt_client_id, "clean_session": True, "protocol": mqtt.MQTTv311}
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, **kwargs)
    except (AttributeError, TypeError):
        client = mqtt.Client(**kwargs)
    if args.mqtt_user:
        client.username_pw_set(args.mqtt_user, args.mqtt_password)

    if args.mock:
        def on_mock(_c, _u, msg):
            if msg.topic != CMD_TOPIC:
                return
            payload = msg.payload.decode("utf-8", errors="replace").strip()
            if not payload.startswith("{"):
                return
            compact = payload[: payload.rfind("}") + 1]
            client.publish(ACK_TOPIC, _ack_from_cmd(compact), qos=1, retain=False)
        client.on_message = on_mock
    else:
        client.on_message = _on_message

    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.subscribe(CMD_TOPIC, qos=1)
    _client = client
    client.loop_start()
    threading.Thread(target=_reader, args=(args.mock,), daemon=True).start()
    print(
        f"桥接已开  {CMD_TOPIC} -> 串口  |  串口 -> {ACK_TOPIC}  |  "
        f"{'MOCK' if args.mock else args.port}"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    _stop.set()
    client.loop_stop()
    client.disconnect()
    if _ser is not None:
        _ser.close()


if __name__ == "__main__":
    main()
