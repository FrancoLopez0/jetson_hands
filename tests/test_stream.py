import time

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from jetson_hands.config import Settings
from jetson_hands.server import create_app

TOKEN = "a" * 43


class TestPipeline:
    __test__ = False

    def __init__(self, settings, loop, hub):
        self.loop, self.hub = loop, hub
        self.sequence = 0
        self.handle = None

    def start(self):
        self.sequence += 1
        self.hub.publish({"type": "frame", "sequence": self.sequence}, b"jpeg", time.monotonic())
        self.handle = self.loop.call_later(0.02, self.start)

    def close(self):
        if self.handle:
            self.handle.cancel()


class TestTelemetry:
    __test__ = False
    latest = {"type": "telemetry", "sample_id": 1, "power": {"input_w": 7.5}}
    error = None

    async def start(self):
        pass

    async def close(self):
        pass


def client():
    return TestClient(create_app(Settings(token=TOKEN), TestPipeline, TestTelemetry))


def receive_frame(ws):
    while True:
        message = ws.receive_json()
        if message["type"] == "ping":
            ws.send_json({"type": "pong"})
            continue
        assert message["type"] == "frame"
        assert ws.receive_bytes() == b"jpeg"
        return message["sequence"]


@pytest.mark.parametrize("auth", [{"token": "wrong"}, {"token": "á" * 43}, [], {"token": None}])
def test_invalid_auth_never_receives_image(auth):
    with client() as http, http.websocket_connect("/stream") as ws:
        ws.send_json(auth)
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_bytes()
        assert error.value.code == 4401
        assert not http.get("/health").json()["receiver_connected"]


def test_single_receiver_and_slot_release():
    with client() as http:
        with http.websocket_connect("/stream") as first:
            first.send_json({"token": TOKEN})
            assert first.receive_json()["type"] == "ready"
            sequence = receive_frame(first)
            assert http.get("/health").json()["receiver_connected"]
            with http.websocket_connect("/stream") as second:
                second.send_json({"token": TOKEN})
                assert second.receive_json()["type"] == "error"
                with pytest.raises(WebSocketDisconnect) as error:
                    second.receive_json()
                assert error.value.code == 4409
            first.send_json({"type": "ack", "sequence": sequence})
            assert receive_frame(first) > sequence
        deadline = time.monotonic() + 2
        while http.get("/health").json()["receiver_connected"] and time.monotonic() < deadline:
            time.sleep(0.02)
        with http.websocket_connect("/stream") as third:
            third.send_json({"token": TOKEN})
            assert third.receive_json()["type"] == "ready"
            receive_frame(third)


def test_incorrect_ack_releases_receiver():
    with client() as http, http.websocket_connect("/stream") as ws:
        ws.send_json({"token": TOKEN})
        assert ws.receive_json()["type"] == "ready"
        sequence = receive_frame(ws)
        ws.send_json({"type": "ack", "sequence": sequence + 1})
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 4400


def test_health_does_not_expose_recognition_or_token():
    with client() as http:
        response = http.get("/health")
        assert response.status_code == 200
        assert set(response.json()) == {"ready", "receiver_connected"}


def test_slow_receiver_skips_frames_instead_of_building_a_queue():
    with client() as http, http.websocket_connect("/stream") as ws:
        ws.send_json({"token": TOKEN})
        assert ws.receive_json()["type"] == "ready"
        first = receive_frame(ws)
        time.sleep(0.15)
        ws.send_json({"type": "ack", "sequence": first})
        assert receive_frame(ws) > first + 1


def test_unauthenticated_connection_does_not_claim_receiver_slot():
    with client() as http, http.websocket_connect("/stream"):
        with http.websocket_connect("/stream") as authenticated:
            authenticated.send_json({"token": TOKEN})
            assert authenticated.receive_json()["type"] == "ready"
            receive_frame(authenticated)


def test_telemetry_opt_in_shares_the_exclusive_authenticated_socket():
    with client() as http, http.websocket_connect("/stream") as ws:
        ws.send_json({"token": TOKEN, "telemetry": True})
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json() == TestTelemetry.latest
        first = receive_frame(ws)
        ws.send_json({"type": "ack", "sequence": first})
        # No duplicate telemetry sample, and no message inserted before frame bytes.
        assert receive_frame(ws) > first


def test_invalid_token_cannot_read_telemetry():
    with client() as http, http.websocket_connect("/stream") as ws:
        ws.send_json({"token": "wrong", "telemetry": True})
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 4401


def test_telemetry_failure_does_not_interrupt_video():
    with client() as http, http.websocket_connect("/stream") as ws:
        http.app.state.telemetry.latest = None
        http.app.state.telemetry.error = "Sensor unavailable"
        ws.send_json({"token": TOKEN, "telemetry": True})
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json()["type"] == "telemetry_status"
        receive_frame(ws)
