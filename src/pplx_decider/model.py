"""The ``Decider``: checkpoint loading, the exact (padded) reference path, and dispatch to the optional fast path."""

from __future__ import annotations

import math
import string
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from .protocol import (
    LEGACY_MAX_PIXELS,
    MAX_OPTIONS,
    MIN_PIXELS,
    PROTOCOL,
    Encoded,
    Encoder,
    answer,
    left_pad,
    load_json,
)
from .types import Request, Response

if TYPE_CHECKING:
    from .fast import FastPath

#: Default per-sequence token limit. Longer requests are split into question chunks (each with the full state).
DEFAULT_MAX_LENGTH = 262_144  # backbone limit (max_position_embeddings); training used sequences up to about 32k tokens


@dataclass(frozen=True)
class FastOptions:
    """Settings of the packed fast path (requires CUDA and ``flash-linear-attention``).

    compile:      ``None`` (eager), ``"sub"`` (norms, MLPs and convolution) or ``"layer"`` (whole decoder layers).
    cuda_graphs:  pad every pack to one of ``buckets`` tokens with filler sequences and replay CUDA graphs. Needs
                  ``compile="layer"``. Compiles once per bucket (a few minutes in total, see ``Decider.warmup``).
    buckets:      static pack sizes for ``cuda_graphs``; the largest is also the packing budget.
    token_budget: tokens per packed forward pass without ``cuda_graphs``.
    prune_last:   compute the last layer's attention output and MLP only at the readout markers.
    fp8:          rowwise-scaled FP8 matmuls in the decoder layers. Faster but changes probabilities slightly (see the
                  README). Applies to every path of this ``Decider`` instance.
    prefetch:     tokenize in a background thread while the GPU runs.
    """

    compile: Literal["sub", "layer"] | None = "layer"
    cuda_graphs: bool = True
    buckets: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)
    token_budget: int = 32768
    prune_last: bool = True
    fp8: bool = False
    prefetch: bool = True

    def budget(self) -> int:
        return max(self.buckets) if self.cuda_graphs else self.token_budget


def enable_noncausal_full_attention(text_model: torch.nn.Module) -> None:
    """Full-attention layers attend to the whole (non-padding) sequence; linear-attention layers stay causal."""

    def mask_inputs(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        if args:
            raise ValueError("The noncausal mask hook requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Noncausal classification does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if isinstance(padding, dict):  # already converted
            return args, kwargs
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def _resolve(path_or_repo: str | Path, revision: str | None, token: str | bool | None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision, token=token))


class Decider:
    """Answers multi-question decision requests with a protocol-v3 joint checkpoint.

    >>> decider = Decider.from_pretrained("perplexity-ai/<repo>", device="cuda")
    >>> decider.decide({"state": "...", "questions": {"route": {"type": "choice", "criteria": {...}}}})
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cuda",
        fast: bool | FastOptions = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        max_pixels: int | None = None,
        batch_tokens: int = 32768,
        cpu_threads: int = 8,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        protocol = load_json(checkpoint / "joint_config.json") if (checkpoint / "joint_config.json").exists() else {}
        if protocol != PROTOCOL:
            raise ValueError("This checkpoint does not declare the joint inline-readout protocol (version 3)")
        config = load_json(checkpoint / "decision_config.json")
        if config.get("attention_mode") != "noncausal_full_attention" or config.get("pooling", "last") != "last":
            raise ValueError("Expected a noncausal checkpoint with per-position readouts")
        self.temperature = float(config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Saved temperature must be positive and finite")
        self.codes: list[str] = list(config["codes"])
        self.max_length = max_length
        self.batch_tokens = batch_tokens
        self.max_pixels = int(max_pixels or config.get("max_pixels") or LEGACY_MAX_PIXELS)

        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.enable_cudnn_sdp(False)  # the evaluated SDPA kernel choice
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": self.max_pixels}
        tokenizer = self.processor.tokenizer
        if len(self.codes) != MAX_OPTIONS or any(
            tokenizer.encode(code, add_special_tokens=False) != [token_id]
            for code, token_id in zip(self.codes, config["token_ids"], strict=True)
        ):
            raise ValueError("Checkpoint answer codes differ from its tokenizer")
        if any(not set(code) <= set(string.ascii_uppercase) for code in self.codes):
            raise ValueError("Unexpected answer codes")
        self.encoder = Encoder(self.processor, self.codes)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.backbone = Qwen3_5Model.from_pretrained(str(checkpoint), dtype=dtype, attn_implementation="sdpa")
        hidden = self.backbone.config.text_config.hidden_size
        self.readout = torch.nn.Linear(hidden, MAX_OPTIONS, bias=False, dtype=dtype)
        self.readout.load_state_dict(load_file(str(checkpoint / "readout.safetensors")))
        enable_noncausal_full_attention(self.backbone.language_model)
        self.backbone.requires_grad_(False).eval().to(self.device)
        self.readout.requires_grad_(False).eval().to(self.device)

        self.fast: FastPath | None = None
        if fast:
            from .fast import FastPath

            self.fast = FastPath(self, fast if isinstance(fast, FastOptions) else FastOptions())

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        device: str = "cuda",
        fast: bool | FastOptions = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        max_pixels: int | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> Decider:
        """Load a local checkpoint directory or a Hugging Face repository.

        fast:       ``False`` for the reference path (padded batches, exact evaluated computation); ``True`` or a
                    ``FastOptions`` for the packed path (same results up to bf16 noise, several times faster).
        max_length: per-sequence token limit; longer requests are split into question chunks, see the README.
        max_pixels: image pixel cap (default: the checkpoint's ``max_pixels``, else 262,144).
        """
        return cls(_resolve(path_or_repo, revision, token), device=device, fast=fast, max_length=max_length,
                   max_pixels=max_pixels, **kwargs)

    # ------------------------------------------------------------------ public API

    @overload
    def decide(self, requests: Request, *, temperature: float | None = None) -> Response: ...
    @overload
    def decide(self, requests: Sequence[Request], *, temperature: float | None = None) -> list[Response]: ...

    def decide(
        self, requests: Request | Sequence[Request], *, temperature: float | None = None
    ) -> Response | list[Response]:
        """Answer one request (returns a ``Response``) or a list of requests (returns a list, same order).

        Each response holds one answer per question and the token usage. ``temperature`` overrides the calibrated
        temperature saved with the checkpoint (leave it unset for calibrated probabilities).
        """
        single = isinstance(requests, dict)
        batch: list[Request] = [requests] if single else list(requests)  # type: ignore[list-item]
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Temperature must be positive and finite")
        responses = [self._respond(request, logits, chunks, scale) for request, logits, chunks in self._run(batch)]
        return responses[0] if single else responses

    def logits(self, requests: Sequence[Request]) -> list[dict[str, list[float]]]:
        """Raw readout logits per question (valid options only, before temperature). Useful for ranking."""
        return [
            {key: values for key, values in logits.items()} for _, logits, _ in self._run(list(requests))
        ]

    def warmup(self) -> None:
        """Compile the fast path for every bucket now instead of on first use (no-op for the reference path)."""
        if self.fast is not None:
            self.fast.warmup()

    # ------------------------------------------------------------------ internals

    def _respond(
        self, request: Request, logits: dict[str, list[float]], chunks: list[Encoded], scale: float
    ) -> Response:
        answers = {}
        for key, question in request["questions"].items():
            values = torch.tensor(logits[key], dtype=torch.float32)
            answers[key] = answer(question, (values / scale).softmax(-1).tolist())
        return {
            "answers": answers,
            "usage": {"input_tokens": sum(c.length for c in chunks), "output_tokens": 0, "sequences": len(chunks)},
        }

    def _run(self, requests: list[Request]) -> Iterator[tuple[Request, dict[str, list[float]], list[Encoded]]]:
        if self.fast is not None:
            yield from self.fast.run(requests)
            return
        plans = [self.encoder.plan(request, self.max_length) for request in requests]
        flat = [(index, chunk) for index, plan in enumerate(plans) for chunk in plan]
        results: dict[tuple[int, str], list[float]] = {}
        for group in self._batches(flat):
            for (index, chunk), rows in zip(group, self.reference_logits([c for _, c in group]), strict=True):
                for key, values in zip(chunk.keys, rows, strict=True):
                    results[index, key] = values
        for index, (request, plan) in enumerate(zip(requests, plans, strict=True)):
            yield request, {key: results[index, key] for key in request["questions"]}, plan

    def _batches(self, flat: list[tuple[int, Encoded]]) -> Iterable[list[tuple[int, Encoded]]]:
        """Length-sorted batches whose padded size stays within ``batch_tokens`` (always at least one sequence)."""
        order = sorted(flat, key=lambda item: item[1].length)
        group: list[tuple[int, Encoded]] = []
        for item in order:
            if group and (len(group) + 1) * item[1].length > self.batch_tokens:
                yield group
                group = []
            group.append(item)
        if group:
            yield group

    @torch.inference_mode()
    def reference_logits(self, chunks: Sequence[Encoded]) -> list[list[list[float]]]:
        """Exact evaluated computation: one left-padded batch through the unmodified Transformers backbone."""
        inputs = {key: value.to(self.device) for key, value in left_pad(chunks, self.encoder.pad_id).items()}
        hidden = self.backbone(**inputs, use_cache=False).last_hidden_state
        offsets = [inputs["input_ids"].shape[1] - chunk.length for chunk in chunks]
        rows = torch.cat([torch.full((len(c.keys),), i) for i, c in enumerate(chunks)]).to(self.device)
        cols = torch.cat([c.markers + offsets[i] for i, c in enumerate(chunks)]).to(self.device)
        logits = self.readout(hidden[rows, cols]).float().cpu()
        out, cursor = [], 0
        for chunk in chunks:
            out.append([logits[cursor + j, :count].tolist() for j, count in enumerate(chunk.counts)])
            cursor += len(chunk.keys)
        return out
