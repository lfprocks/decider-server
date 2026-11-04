# decider-server

A small HTTP API for [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2), a decision model. It answers typed questions about a state with calibrated probabilities. It does not generate text and has no chat API.

## API

`POST /v1/decide`, with `Authorization: Bearer <DECIDER_API_KEY>`:

```bash
curl -s "$URL/v1/decide" \
  -H "Authorization: Bearer $DECIDER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "My integration keeps failing with a 401 since we rotated our API key.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Charges and refunds", "support": "Technical errors", "sales": "Buying"}
      }
    }
  }'
```

The request and response follow the [model card](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2#request-schema), with two differences: `images` must be `data:image/...;base64` URLs, because the library opens any other string as a local file, and `temperature` can't be set. The card recommends one question per request for this model unless the questions are facets of one decision.

| Status | Meaning |
|---|---|
| 200 | Answered |
| 400 | Invalid JSON, schema or image |
| 401 | Missing or wrong API key |
| 413 | Body or input is too long. Inputs are never truncated. |
| 503 | Queue is full. See `Retry-After`. |

Requests run one at a time on the GPU. The server doesn't log request content.

`GET /healthz` answers once the model has loaded and passed a test decision. Prometheus metrics are on `:8081/metrics`.

## Configuration

| Variable | Default | |
|---|---|---|
| `DECIDER_API_KEY` | required | Bearer token |
| `DECIDER_MAX_LENGTH` | `8192` | Tokens per sequence. The model was trained up to about 32k. More uses more GPU memory. |
| `DECIDER_MAX_IMAGES` | `8` | Images per request |
| `DECIDER_MAX_BODY_BYTES` | `33554432` | Request body limit |
| `DECIDER_MAX_PENDING` | `16` | Running plus waiting requests before 503 |
| `DECIDER_DEVICE` | `cuda` | Torch device |
| `HF_HOME` | `/models` | Weights cache. Use a volume: the first start downloads about 8.5 GiB. |

Needs an NVIDIA driver that supports CUDA 13.0. Measured with one request at a time: 9.7 GiB of GPU memory at 2k tokens, 11.7 GiB at 7k, and a 6.3 GiB peak in host memory during loading. `/tmp` must be writable and allow execution, because torch compiles a small helper there on first use.

## Model and library

`src/pplx_decider` is Perplexity's inference code, unmodified, vendored from the model revision pinned in `DECIDER_MODEL_REVISION` under Apache-2.0 (see its `LICENSE` and `NOTICE`). Change the revision and the vendored code together.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```

Tests use a stub in place of the model. Pushing a `v*` tag publishes the image to GHCR.
