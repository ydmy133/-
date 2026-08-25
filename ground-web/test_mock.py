"""阶段 1 mock 协议验收：不插硬件。"""

from __future__ import annotations

import server as gw
from fastapi.testclient import TestClient

CMD = "boat/boat01/cmd"


def _drain_until(ws, pred, limit=12):
    last = None
    for _ in range(limit):
        last = ws.receive_json()
        if pred(last):
            return last
    raise AssertionError(f"未等到预期消息，最后一条: {last}")


def test_mock_speed_and_stop() -> None:
    gw._mock = True
    gw._active_ws = None
    gw._close_serial()
    with TestClient(gw.app) as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["op"] == "hello"
            assert hello["mock"] is True
            assert hello.get("mode") in ("mock", None) or hello["mock"] is True
            assert hello["cmdTopic"] == CMD

            ws.send_json({"op": "open", "port": "MOCK"})
            st = _drain_until(ws, lambda m: m.get("op") == "status")
            assert st["serial"] is True

            ws.send_json(
                {
                    "op": "publish",
                    "topic": CMD,
                    "payload": '{"mode":"speed","T":20,"Y":8}',
                }
            )
            ack = _drain_until(ws, lambda m: m.get("op") == "message")
            assert ack["payload"] == "OK T=200 Y=80"

            ws.send_json(
                {"op": "publish", "topic": CMD, "payload": '{"mode":"stop"}'}
            )
            ack = _drain_until(ws, lambda m: m.get("op") == "message")
            assert ack["payload"] == "OK T=0 Y=0"


def test_second_page_rejected() -> None:
    gw._mock = True
    gw._active_ws = None
    with TestClient(gw.app) as client:
        with client.websocket_connect("/ws") as ws1:
            assert ws1.receive_json()["op"] == "hello"
            with client.websocket_connect("/ws") as ws2:
                msg = ws2.receive_json()
                assert msg["op"] == "rejected"


if __name__ == "__main__":
    test_mock_speed_and_stop()
    test_second_page_rejected()
    print("mock ok")
