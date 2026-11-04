import base64
import io
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from decider_server.main import Settings, create_app
from pplx_decider import LengthError, RequestError

AUTH = {"Authorization": "Bearer test-key"}
QUESTION = {"check": {"type": "noul", "instructions": "Is it?"}}


class FakeDecider:
    def __init__(self, fail: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.fail = fail
        self.gate = gate
        self.seen: list[dict[str, Any]] = []

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.gate:
            self.gate.wait(5)
        self.seen.append(request)
        if self.fail:
            raise self.fail
        return {"answers": {"check": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 7}}


def client(decider: FakeDecider, **overrides: Any) -> TestClient:
    settings = Settings(api_key="test-key", **overrides)
    return TestClient(create_app(settings, load=lambda _: decider), raise_server_exceptions=False)


def png() -> str:
    out = io.BytesIO()
    Image.new("RGB", (4, 4)).save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def test_answers_a_decision() -> None:
    with client(FakeDecider()) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION})
    assert r.status_code == 200
    assert r.json()["answers"]["check"]["noul"] == 0.9


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "test-key"}])
def test_rejects_bad_credentials_without_deciding(headers: dict[str, str]) -> None:
    decider = FakeDecider()
    with client(decider) as c:
        r = c.post("/v1/decide", headers=headers, json={"state": "s", "questions": QUESTION})
    assert r.status_code == 401
    assert decider.seen == []


def test_empty_api_key_refuses_to_start() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        create_app(Settings(api_key=""), load=lambda _: FakeDecider())


def test_health_needs_no_key_and_the_decision_docs_are_off() -> None:
    with client(FakeDecider()) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404


@pytest.mark.parametrize("image", ["/etc/passwd", "../secret.png", "http://example.invalid/a.png", "data:text/plain;base64,aGk="])
def test_never_passes_a_non_inline_image_to_the_library(image: str) -> None:
    decider = FakeDecider()
    with client(decider) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION, "images": [image]})
    assert r.status_code == 400
    assert decider.seen == []


def test_decodes_inline_images_before_the_library_sees_them() -> None:
    decider = FakeDecider()
    with client(decider) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION, "images": [png()]})
    assert r.status_code == 200
    assert isinstance(decider.seen[0]["images"][0], Image.Image)


@pytest.mark.parametrize("image", ["data:image/png;base64,!!!", "data:image/png;base64,aGVsbG8="])
def test_undecodable_image_is_a_client_error(image: str) -> None:
    with client(FakeDecider()) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION, "images": [image]})
    assert r.status_code == 400


def test_too_many_images() -> None:
    with client(FakeDecider(), max_images=1) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION, "images": [png(), png()]})
    assert r.status_code == 400


def test_oversized_body() -> None:
    with client(FakeDecider(), max_body_bytes=64) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "x" * 200, "questions": QUESTION})
    assert r.status_code == 413


@pytest.mark.parametrize(
    "body",
    [
        {"questions": QUESTION},
        {"state": "s"},
        {"state": 1, "questions": QUESTION},
        {"state": "s", "questions": QUESTION, "temperature": 2},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": "nope"}}},
    ],
)
def test_schema_errors(body: dict[str, Any]) -> None:
    with client(FakeDecider()) as c:
        assert c.post("/v1/decide", headers=AUTH, json=body).status_code == 400


def test_invalid_json() -> None:
    with client(FakeDecider()) as c:
        assert c.post("/v1/decide", headers=AUTH, content=b"{").status_code == 400


@pytest.mark.parametrize(("fail", "status"), [(RequestError("bad"), 400), (LengthError("long"), 413)])
def test_library_errors_map_to_client_errors(fail: Exception, status: int) -> None:
    with client(FakeDecider(fail=fail)) as c:
        assert c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION}).status_code == status


def test_unexpected_error_hides_its_message(caplog: pytest.LogCaptureFixture) -> None:
    with client(FakeDecider(fail=RuntimeError("private state text"))) as c:
        r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION})
    assert r.status_code == 500
    assert "private state text" not in r.text
    assert "private state text" not in caplog.text


def test_sheds_load_when_the_queue_is_full() -> None:
    gate = threading.Event()
    with client(FakeDecider(gate=gate), max_pending=1) as c:
        first: dict[str, int] = {}

        def send() -> None:
            first["status"] = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION}).status_code

        t = threading.Thread(target=send)
        t.start()
        try:
            for _ in range(100):
                r = c.post("/v1/decide", headers=AUTH, json={"state": "s", "questions": QUESTION})
                if r.status_code == 503:
                    break
                threading.Event().wait(0.02)
            assert r.status_code == 503
            assert r.headers["Retry-After"] == "5"
        finally:
            gate.set()
            t.join()
    assert first["status"] == 200
