"""HTTP API for pplx-decider Flash: one authenticated decision endpoint in front of the vendored library."""

from __future__ import annotations

import asyncio
import binascii
import hmac
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pplx_decider import Decider, LengthError, RequestError
from pplx_decider.protocol import open_image

log = logging.getLogger("decider_server")

REQUESTS = Counter("decider_requests_total", "Decision requests by outcome", ["outcome"])
SECONDS = Histogram(
    "decider_request_seconds",
    "Time to answer a successful request, including queueing",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
INPUT_TOKENS = Counter("decider_input_tokens_total", "Input tokens in answered requests")
PENDING = Gauge("decider_pending_requests", "Requests running or waiting for the GPU")

# Decoded images are capped at the model's own limit later; this bounds the memory used to decode one.
# Pillow raises DecompressionBombError at twice this value.
Image.MAX_IMAGE_PIXELS = 20_000_000
DATA_IMAGE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DECIDER_", protected_namespaces=())

    api_key: SecretStr
    model_repo: str = "perplexity-ai/pplx-decider-flash-v1.2"
    # Must stay the revision src/pplx_decider was vendored from.
    model_revision: str = "9ff7cf9838e59df461c276b7699d56b244d2af58"
    device: str = "cuda"
    max_length: int = 8192
    max_images: int = 8
    max_body_bytes: int = 32 * 1024 * 1024
    max_pending: int = 16
    port: int = 8080
    metrics_port: int = 8081


class BadImage(Exception):
    """An inline image could not be decoded."""


class Decisions(Protocol):
    def decide(self, requests: dict[str, Any]) -> dict[str, Any]: ...


class DecideRequest(BaseModel):
    """The library's request schema, narrowed: images must be inline, because it opens any other string as a path."""

    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]]
    images: list[str] = []

    @field_validator("images")
    @classmethod
    def inline_images_only(cls, images: list[str]) -> list[str]:
        for image in images:
            if not DATA_IMAGE.match(image):
                raise ValueError("images must be data:image/...;base64 URLs")
        return images

    @field_validator("questions")
    @classmethod
    def criteria_shape(cls, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        for key, question in questions.items():
            criteria = question.get("criteria")
            if question.get("type") == "noul" and not (criteria is None or isinstance(criteria, dict)):
                raise ValueError(f"question {key!r}: noul criteria must map 'true' and 'false' to descriptions")
        return questions


def load_decider(settings: Settings) -> Decisions:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        settings.model_repo,
        revision=settings.model_revision,
        ignore_patterns=["source/*", "training/*", "evaluation/*"],
    )
    decider = Decider.from_pretrained(path, device=settings.device, max_length=settings.max_length)
    # Fail at startup, not on the first caller, if the model cannot run on this GPU.
    decider.decide({"state": "ok", "questions": {"check": {"type": "noul", "instructions": "Is this a check?"}}})
    return decider


def error(status: int, message: str, outcome: str, headers: dict[str, str] | None = None) -> JSONResponse:
    REQUESTS.labels(outcome).inc()
    return JSONResponse({"error": message}, status_code=status, headers=headers)


async def read_limited(request: Request, limit: int) -> bytes | None:
    """The body, or None once it passes the limit. Chunked requests have no Content-Length to check first."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def create_app(settings: Settings, load: Callable[[Settings], Decisions] = load_decider) -> FastAPI:
    # One thread: requests run one at a time on the GPU, and the rest wait here.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decider")
    state: dict[str, Any] = {"pending": 0}
    key = settings.api_key.get_secret_value().encode()
    if not key:
        raise ValueError("DECIDER_API_KEY must not be empty")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Blocks until loaded, so /healthz only answers once the model is ready.
        state["decider"] = await asyncio.get_running_loop().run_in_executor(pool, load, settings)
        log.info("model ready: %s@%s", settings.model_repo, settings.model_revision)
        yield
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="decider-server", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/decide")
    async def decide(request: Request) -> JSONResponse:
        supplied = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(supplied, b"Bearer " + key):
            return error(401, "invalid or missing API key", "unauthorized", {"WWW-Authenticate": "Bearer"})

        body = await read_limited(request, settings.max_body_bytes)
        if body is None:
            return error(413, f"body is over {settings.max_body_bytes} bytes", "too_large")
        try:
            parsed = DecideRequest.model_validate(json.loads(body))
        except json.JSONDecodeError:
            return error(400, "body is not valid JSON", "invalid")
        except ValidationError as e:
            details = [{"field": ".".join(map(str, x["loc"])), "message": x["msg"]} for x in e.errors()]
            REQUESTS.labels("invalid").inc()
            return JSONResponse({"error": "invalid request", "details": details}, status_code=400)
        if len(parsed.images) > settings.max_images:
            return error(400, f"at most {settings.max_images} images are allowed", "invalid")

        if state["pending"] >= settings.max_pending:
            return error(503, "busy, try again shortly", "busy", {"Retry-After": "5"})

        def run() -> dict[str, Any]:
            try:
                images = [open_image(image) for image in parsed.images]
            except (binascii.Error, OSError, Image.DecompressionBombError) as e:
                raise BadImage from e
            return state["decider"].decide(parsed.model_dump() | {"images": images})

        started = time.monotonic()
        state["pending"] += 1
        PENDING.set(state["pending"])
        try:
            response = await asyncio.get_running_loop().run_in_executor(pool, run)
        except LengthError as e:
            return error(413, str(e), "too_long")
        except RequestError as e:
            return error(400, str(e), "invalid")
        except BadImage:
            return error(400, "an image could not be decoded, or is too large", "invalid")
        except Exception:  # noqa: BLE001
            # The message can quote request content, so it stays out of the response and the log.
            log.error("decision failed with an unexpected error")
            return error(500, "internal error", "error")
        finally:
            state["pending"] -= 1
            PENDING.set(state["pending"])

        REQUESTS.labels("ok").inc()
        SECONDS.observe(time.monotonic() - started)
        INPUT_TOKENS.inc(response.get("usage", {}).get("input_tokens", 0))
        return JSONResponse(response)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings()
    start_http_server(settings.metrics_port)
    uvicorn.run(create_app(settings), host="0.0.0.0", port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
