# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM python:3.12-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
# Dependencies first, for a cached layer. Torch comes from the cu130 index per uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models
# Torch's Triton kernels compile a small helper with a C compiler on first use.
# apt pins are not stable on a rolling base
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /models \
    && chown 10001:10001 /models
COPY --from=build /opt/venv /opt/venv
USER 10001:10001
EXPOSE 8080 8081
ENTRYPOINT ["python", "-m", "decider_server.main"]
