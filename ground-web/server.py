"""本机网页网关：默认写串口；--mqtt 时发到 Broker（阶段 2）。主题仍是 cmd / ack。"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:  # 仅 --mqtt 需要；阶段 1 mock / 串口可不装
    mqtt = None  # type: ignore

import serial
import serial.tools.list_ports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CMD_TOPIC = "boat/boat01/cmd"
ACK_TOPIC = "boat/boat01/ack"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    yield
    _close_serial()
    _close_mqtt()


app = FastAPI(title="ground-web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_mock = False
_baud = 115200
_mqtt_enabled = False
_mqtt_host = "127.0.0.1"
_mqtt_port = 1883
_mqtt_user = ""
_mqtt_password = ""
_mqtt_client_id = "boat01-webgw"
_mqtt: Optional[object] = None
_ws_lock = asyncio.Lock()
_active_ws: Optional[WebSocket] = None
_ser: Optional[serial.Serial] = None
_ser_lock = threading.Lock()
_reader_stop = threading.Event()
_loop: Optional[asyncio.AbstractEventLoop] = None
_last_cmd_log = ""
_tx_q: queue.Queue = queue.Queue()
_ser_thread: Optional[threading.Thread] = None
_io_stats = {"tx": 0, "rx": 0}


def list_com_ports() -> list:
    ports = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").strip()
        label = f"{p.device}  {desc}" if desc else p.device
        ports.append({"device": p.device, "label": label})
    return ports


def _notify(obj: dict) -> None:
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(obj), _loop)


def _close_serial() -> None:
    global _ser, _ser_thread
    _reader_stop.set()
    while True:
        try:
            _tx_q.get_nowait()
        except queue.Empty:
            break
    ser = None
    with _ser_lock:
        ser = _ser
        _ser = None
    if ser is not None:
        try:
            cancel = getattr(ser, "cancel_read", None)
            if callable(cancel):
                cancel()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
    th = _ser_thread
    _ser_thread = None
    if th is not None and th.is_alive() and th is not threading.current_thread():
        th.join(timeout=1.0)


def _serial_worker() -> None:
    buf = bytearray()
    last_rx = time.monotonic()
    while not _reader_stop.is_set():
        with _ser_lock:
            ser = _ser
        if ser is None or not ser.is_open:
            break
        try:
            while True:
                text = _tx_q.get_nowait()
                data = str(text).encode("utf-8")
                if not data.endswith(b"\n"):
                    data += b"\r\n"
                ser.write(data)
        except queue.Empty:
            pass
        except Exception as exc:
            _notify({"op": "log", "text": f"串口写失败: {exc}"})
        try:
            chunk = ser.read(256)
        except Exception:
            chunk = b""
        if chunk:
            last_rx = time.monotonic()
            _io_stats["rx"] += len(chunk)
            buf.extend(chunk)
        elif buf and (time.monotonic() - last_rx) > 0.15:
            line = buf.decode("utf-8", errors="replace").strip()
            buf.clear()
            if line:
                _notify({"op": "message", "topic": ACK_TOPIC, "payload": line})
            continue
        else:
            continue
        while True:
            n_lf = buf.find(b"\n")
            n_cr = buf.find(b"\r")
            if n_lf < 0 and n_cr < 0:
                if len(buf) > 256:
                    leftover = buf.decode("utf-8", errors="replace").strip()
                    buf.clear()
                    if leftover:
                        _notify({"op": "message", "topic": ACK_TOPIC, "payload": leftover})
                break
            cuts = [i for i in (n_lf, n_cr) if i >= 0]
            nl = min(cuts)
            line = buf[:nl].decode("utf-8", errors="replace").strip()
            skip = nl + 1
            if skip < len(buf) and buf[nl] == 0x0D and buf[skip] == 0x0A:
                skip += 1
            del buf[:skip]
            if line:
                _notify({"op": "message", "topic": ACK_TOPIC, "payload": line})


def _open_serial(port: str) -> None:
    global _ser, _ser_thread
    _close_serial()
    _reader_stop.clear()
    _io_stats["tx"] = 0
    _io_stats["rx"] = 0
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = _baud
    ser.timeout = 0.05
    ser.write_timeout = 0.5
    ser.dsrdtr = False
    ser.rtscts = False
    ser.xonxoff = False
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass
    try:
        ser.exclusive = True
    except Exception:
        pass
    ser.open()
    # 开串口常会脉冲 DTR，STM32 复位需要等启动后再收发
    time.sleep(0.6)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    with _ser_lock:
        _ser = ser
    _ser_thread = threading.Thread(target=_serial_worker, name="uart-io", daemon=True)
    _ser_thread.start()


def _write_serial(text: str, flush: bool = False) -> None:
    data = str(text).encode("utf-8")
    if not data.endswith(b"\n"):
        data += b"\r\n"
    with _ser_lock:
        opened = _ser is not None and _ser.is_open
    if not opened:
        raise RuntimeError("串口未打开")
    if flush:
        while True:
            try:
                _tx_q.get_nowait()
            except queue.Empty:
                break
    _tx_q.put(text)
    _io_stats["tx"] += len(data)


def _ack_from_cmd(payload: str) -> str:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "OK T=0 Y=0"
    mode = str(obj.get("mode", ""))
    if mode == "stop":
        return "OK T=0 Y=0"
    t = int(obj.get("T", 0))
    y = int(obj.get("Y", 0))
    t = max(-100, min(100, t))
    y = max(-100, min(100, y))
    return f"OK T={t * 10} Y={y * 10}"


async def _send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


async def _broadcast(obj: dict) -> None:
    ws = _active_ws
    if ws is None:
        return
    try:
        await _send(ws, obj)
    except Exception:
        pass


def _transport_mode() -> str:
    if _mqtt_enabled:
        return "mqtt"
    if _mock:
        return "mock"
    return "serial"


def _close_mqtt() -> None:
    global _mqtt
    client = _mqtt
    _mqtt = None
    if client is None:
        return
    try:
        client.loop_stop()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        client.disconnect()  # type: ignore[attr-defined]
    except Exception:
        pass


def _on_mqtt_message(_client, _userdata, msg) -> None:
    if msg.topic != ACK_TOPIC:
        return
    payload = msg.payload.decode("utf-8", errors="replace").strip()
    if not payload or _loop is None:
        return
    asyncio.run_coroutine_threadsafe(
        _broadcast({"op": "message", "topic": ACK_TOPIC, "payload": payload}),
        _loop,
    )


def _ensure_mqtt() -> None:
    global _mqtt
    if mqtt is None:
        raise RuntimeError("未安装 paho-mqtt，请执行: pip install paho-mqtt")
    if _mqtt is not None:
        return
    kwargs = {
        "client_id": _mqtt_client_id,
        "clean_session": True,
        "protocol": mqtt.MQTTv311,
    }
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            **kwargs,
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(**kwargs)
    if _mqtt_user:
        client.username_pw_set(_mqtt_user, _mqtt_password)
    client.on_message = _on_mqtt_message
    client.connect(_mqtt_host, _mqtt_port, keepalive=60)
    client.subscribe(ACK_TOPIC, qos=1)
    client.loop_start()
    _mqtt = client


def _write_mqtt(text: str) -> None:
    if _mqtt is None:
        raise RuntimeError("MQTT 未连接")
    info = _mqtt.publish(CMD_TOPIC, text, qos=1, retain=False)
    wait = getattr(info, "wait_for_publish", None)
    if callable(wait):
        wait(2)


def _send_cmd(compact: str, flush: bool = False) -> Optional[str]:
    """写出 cmd。mock 时返回模拟 ack，否则 None。"""
    global _last_cmd_log
    if compact != _last_cmd_log:
        print("CMD", compact, flush=True)
        _last_cmd_log = compact
    if _mock:
        return _ack_from_cmd(compact)
    if _mqtt_enabled:
        _ensure_mqtt()
        _write_mqtt(compact)
        return None
    is_stop = '"stop"' in compact
    _write_serial(compact, flush=flush or is_stop)
    return None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/ports")
async def api_ports() -> dict:
    return {"ports": list_com_ports(), "mock": _mock}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global _active_ws
    await ws.accept()
    async with _ws_lock:
        if _active_ws is not None:
            await _send(ws, {"op": "rejected", "reason": "已有操控页在线，请只开一个标签"})
            await ws.close(code=4000)
            return
        _active_ws = ws
    await _send(
        ws,
        {
            "op": "hello",
            "cmdTopic": CMD_TOPIC,
            "ackTopic": ACK_TOPIC,
            "mock": _mock,
            "mode": _transport_mode(),
            "ports": list_com_ports(),
        },
    )
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, {"op": "log", "text": "网关收到非 JSON"})
                continue
            op = msg.get("op")
            if op == "list_ports":
                await _send(ws, {"op": "ports", "ports": list_com_ports(), "mock": _mock})
            elif op == "open":
                port = str(msg.get("port", "")).strip()
                try:
                    if _mock:
                        await _send(ws, {"op": "status", "serial": True, "mock": True, "port": "MOCK", "mode": "mock"})
                    elif _mqtt_enabled:
                        _ensure_mqtt()
                        await _send(
                            ws,
                            {
                                "op": "status",
                                "serial": True,
                                "mock": False,
                                "mode": "mqtt",
                                "port": f"mqtt://{_mqtt_host}:{_mqtt_port}",
                            },
                        )
                    else:
                        if not port:
                            raise RuntimeError("未选择 COM 口")
                        await asyncio.to_thread(_open_serial, port)
                        await _send(ws, {"op": "status", "serial": True, "mock": False, "mode": "serial", "port": port})
                except Exception as exc:
                    await _send(ws, {"op": "status", "serial": False, "mock": _mock, "mode": _transport_mode(), "error": str(exc)})
            elif op == "close":
                _close_serial()
                if _mqtt_enabled:
                    _close_mqtt()
                await _send(ws, {"op": "status", "serial": False, "mock": _mock, "mode": _transport_mode()})
            elif op == "publish":
                topic = str(msg.get("topic", ""))
                payload = str(msg.get("payload", "")).strip()
                if topic != CMD_TOPIC:
                    await _send(ws, {"op": "log", "text": f"忽略主题 {topic}"})
                    continue
                if not payload.startswith("{") or "}" not in payload:
                    await _send(ws, {"op": "log", "text": "payload 必须是单行 JSON"})
                    continue
                compact = payload[: payload.rfind("}") + 1]
                try:
                    ack = await asyncio.to_thread(
                        _send_cmd, compact, bool(msg.get("flush"))
                    )
                    await _send(
                        ws,
                        {
                            "op": "sent",
                            "payload": compact,
                            "tx": _io_stats["tx"],
                            "rx": _io_stats["rx"],
                        },
                    )
                    if ack is not None:
                        await _send(ws, {"op": "message", "topic": ACK_TOPIC, "payload": ack})
                except Exception as exc:
                    await _send(ws, {"op": "log", "text": f"发送失败: {exc}"})
            else:
                await _send(ws, {"op": "log", "text": f"未知 op: {op}"})
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if _active_ws is ws:
                _active_ws = None
        try:
            _send_cmd('{"mode":"stop"}')
        except Exception:
            pass
        if not _mqtt_enabled:
            _close_serial()


def main() -> None:
    global _mock, _baud, _mqtt_enabled, _mqtt_host, _mqtt_port
    global _mqtt_user, _mqtt_password, _mqtt_client_id
    parser = argparse.ArgumentParser(description="无人船网页网关（串口或 MQTT）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mock", action="store_true", help="无硬件，回显 OK")
    parser.add_argument("--mqtt", action="store_true", help="阶段 2：网页仍走本机 WS，网关改发 MQTT")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-user", default="")
    parser.add_argument("--mqtt-password", default="")
    parser.add_argument("--mqtt-client-id", default="boat01-webgw")
    args = parser.parse_args()
    _mock = args.mock
    _baud = args.baud
    _mqtt_enabled = args.mqtt
    _mqtt_host = args.mqtt_host
    _mqtt_port = args.mqtt_port
    _mqtt_user = args.mqtt_user
    _mqtt_password = args.mqtt_password
    _mqtt_client_id = args.mqtt_client_id
    if _mock and _mqtt_enabled:
        raise SystemExit("--mock 与 --mqtt 不要一起用")
    if _mqtt_enabled and mqtt is None:
        raise SystemExit("阶段 2 需要: pip install paho-mqtt")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
