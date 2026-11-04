"""Packed fast path: whole requests concatenated into one padding-free sequence per forward pass.

Each packed unit is a whole request (or, for over-length requests, a chunk of its questions) with one readout
per question marker. Requests stay independent:

- full-attention layers: varlen FlashAttention (``torch.nn.attention.varlen``) with window (-1, -1), i.e.
  noncausal within each request, which is what the checkpoint's padding-masked noncausal SDPA computes;
- linear-attention layers: ``fla``'s chunked gated delta rule with ``cu_seqlens`` (state reset per request) and a
  depthwise causal convolution that does not read across request boundaries;
- positions restart at 0 for every request; image requests get their own M-RoPE positions (``get_rope_index``);
- readouts are gathered at the marker positions only.

Optional speedups (``FastOptions``): torch.compile of sub-modules or whole layers, static pack sizes with CUDA
graphs (filler sequences reach the bucket size; request boundaries reach the eager kernels through a side
channel), last-layer pruning at marker rows, and FP8 matmuls.

The decoder layers are re-expressed here with the checkpoint's own modules and weights; the Transformers model
is not patched, so the reference path of the same ``Decider`` stays exact.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from .protocol import Encoded
from .types import Request

if TYPE_CHECKING:
    from .model import Decider, FastOptions

FILLER_MAX = 1024  # filler sequences that reach a static size are split into chunks of at most this many tokens
_PACK: dict[str, Any] = {}  # boundaries of the current pack; only read by the eager kernels below


def _require_kernels() -> tuple[Any, Any]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "The fast path needs flash-linear-attention (pip install flash-linear-attention); "
            "use fast=False for the reference path"
        ) from error
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as error:  # pragma: no cover
        raise ImportError("The fast path needs torch>=2.10 with torch.nn.attention.varlen") from error
    return chunk_gated_delta_rule, varlen_attn


_chunk_rule, _varlen = None, None


@torch.compiler.disable
def _gated_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> Any:
    out, _ = _chunk_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                         use_qk_l2norm_in_kernel=True, cu_seqlens=_PACK["cu_long"], cu_seqlens_cpu=_PACK["cu_cpu"])
    return out


@torch.compiler.disable
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    # (1, H, T, D) -> (T, H, D); noncausal inside each request, nothing across requests
    q, k, v = (x[0].transpose(0, 1) for x in (q, k, v))
    cu, mx = _PACK["cu"], _PACK["max"]
    out = _varlen(q, k, v, cu, cu, mx, mx, scale=scale, window_size=(-1, -1), enable_gqa=k.shape[1] != q.shape[1])
    return out.unsqueeze(0)  # (1, T, H, D)


def packed_conv(x: torch.Tensor, weight: torch.Tensor, seqpos: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv over (1, C, T) that never reads taps before the start of a request (fp32 sums)."""
    taps = weight.shape[-1]
    xf = x.float()
    out = xf * weight[:, taps - 1, None].float()
    for shift in range(1, taps):
        shifted = F.pad(xf, (shift, 0))[..., : x.shape[-1]]
        out = out + shifted * (seqpos >= shift).float() * weight[:, taps - 1 - shift, None].float()
    return out.to(x.dtype)


def linear_attention(mod: torch.nn.Module, x: torch.Tensor, seqpos: torch.Tensor, conv: Any) -> torch.Tensor:
    """Qwen3_5GatedDeltaNet.forward without cache or padding, on a packed (1, T, D) sequence."""
    _, length, _ = x.shape
    mixed = mod.in_proj_qkv(x).transpose(1, 2)
    z = mod.in_proj_z(x).reshape(1, length, -1, mod.head_v_dim)
    b = mod.in_proj_b(x)
    a = mod.in_proj_a(x)
    if mod.conv1d.bias is not None:
        raise ValueError("Unexpected convolution bias")
    mixed = ACT2FN[mod.activation](conv(mixed, mod.conv1d.weight.squeeze(1), seqpos)).transpose(1, 2)
    query, key, value = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    query = query.reshape(1, length, -1, mod.head_k_dim)
    key = key.reshape(1, length, -1, mod.head_k_dim)
    value = value.reshape(1, length, -1, mod.head_v_dim)
    beta = b.sigmoid()
    g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
    if mod.num_v_heads // mod.num_k_heads > 1:
        query = query.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
        key = key.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
    core = _gated_delta(query, key, value, g, beta)
    core = mod.norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
    return mod.out_proj(core.reshape(1, length, -1))


def full_attention(attn: torch.nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, length, _ = x.shape
    shape = (1, length, -1, attn.head_dim)
    query, gate = torch.chunk(attn.q_proj(x).view(1, length, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(1, length, -1)
    query = attn.q_norm(query.view(shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
    value = attn.v_proj(x).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    out = _attention(query, key, value, attn.scaling)
    return attn.o_proj(out.reshape(1, length, -1).contiguous() * torch.sigmoid(gate))


def decoder_layer(
    layer: torch.nn.Module, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqpos: torch.Tensor,
    conv: Any = packed_conv,
) -> torch.Tensor:
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        x = linear_attention(layer.linear_attn, x, seqpos, conv)
    else:
        x = full_attention(layer.self_attn, x, cos, sin)
    h = h + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def last_layer_at(
    layer: torch.nn.Module, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sel: torch.Tensor,
    cu_q: torch.Tensor, max_q: int, cu_k: torch.Tensor, max_k: int,
) -> torch.Tensor:
    """Last (full-attention) layer with K/V for every token but queries, output and MLP only at rows ``sel``."""
    attn = layer.self_attn
    x = layer.input_layernorm(h)
    length, hd, n = x.shape[1], attn.head_dim, sel.shape[0]
    key = attn.k_norm(attn.k_proj(x).view(1, length, -1, hd)).transpose(1, 2)
    value = attn.v_proj(x).view(1, length, -1, hd).transpose(1, 2)
    query, gate = torch.chunk(attn.q_proj(x[:, sel]).view(1, n, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(1, n, -1)
    query = attn.q_norm(query).transpose(1, 2)
    query, _ = apply_rotary_pos_emb(query, query, cos[:, sel], sin[:, sel])
    _, key = apply_rotary_pos_emb(key, key, cos, sin)
    q, k, v = (t[0].transpose(0, 1).contiguous() for t in (query, key, value))
    out = _varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale=attn.scaling, window_size=(-1, -1),
                  enable_gqa=k.shape[1] != q.shape[1])
    out = attn.o_proj(out.reshape(1, n, -1) * torch.sigmoid(gate))
    hs = h[:, sel] + out
    return hs + layer.mlp(layer.post_attention_layernorm(hs))


class FP8Linear(torch.nn.Module):
    """Rowwise-scaled float8_e4m3fn matmul: per-output-channel weight scales, per-token activation scales."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.data.float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
        self.register_buffer("wq", (weight / scale).to(torch.float8_e4m3fn))
        self.register_buffer("w_scale", scale.t().contiguous())
        self.bias = linear.bias
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        a_scale = flat.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 448.0
        xq = (flat.float() / a_scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(xq, self.wq.t(), scale_a=a_scale, scale_b=self.w_scale, bias=self.bias,
                               out_dtype=torch.bfloat16)
        return out.reshape(*shape[:-1], self.out_features)


def to_fp8(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0 \
                and child.out_features >= 256:
            setattr(module, name, FP8Linear(child))
        else:
            to_fp8(child)


class FastPath:
    def __init__(self, decider: Decider, options: FastOptions) -> None:
        global _chunk_rule, _varlen
        if decider.device.type != "cuda":
            raise ValueError("The fast path needs a CUDA device")
        _chunk_rule, _varlen = _require_kernels()
        if options.cuda_graphs and options.compile != "layer":
            raise ValueError("cuda_graphs=True needs compile='layer'")
        if options.budget() < 1:
            raise ValueError("The packing budget must be positive")
        self.decider = decider
        self.options = options
        self.lm = decider.backbone.language_model
        self.layers = list(self.lm.layers)
        if self.layers[-1].block_type != "full_attention":
            raise ValueError("Last-layer pruning expects a full-attention last layer")
        self.image_token_id = decider.backbone.config.image_token_id
        if options.fp8:
            to_fp8(self.lm.layers)
            torch.cuda.empty_cache()
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 4096)
        self.conv = packed_conv
        self.layer_fn: Any = decoder_layer
        self.graph_fn: Any = None
        if options.compile == "sub":
            self.conv = torch.compile(packed_conv, dynamic=True)
            for layer in self.layers:
                mods = [layer.mlp, layer.input_layernorm, layer.post_attention_layernorm]
                mods += [layer.linear_attn.norm] if layer.block_type == "linear_attention" else \
                    [layer.self_attn.q_norm, layer.self_attn.k_norm]
                for mod in mods:
                    mod.forward = torch.compile(mod.forward, dynamic=True)
        elif options.compile == "layer":
            if options.cuda_graphs:
                self.graph_fn = torch.compile(decoder_layer, mode="reduce-overhead", dynamic=False)
            self.layer_fn = torch.compile(decoder_layer, dynamic=True)
        self.last_fn: Any = last_layer_at if options.compile is None else torch.compile(last_layer_at, dynamic=True)
        self.buckets = tuple(sorted(options.buckets)) if options.cuda_graphs else ()

    # -------------------------------------------------------------- scheduling

    def run(self, requests: list[Request]) -> Iterator[tuple[Request, dict[str, list[float]], list[Encoded]]]:
        """Yield (request, logits per question, chunks) in input order."""
        encoder, max_length = self.decider.encoder, self.decider.max_length
        for request in requests:
            encoder.validate(request)
        budget = self.options.budget()

        def plans() -> Iterator[tuple[int, list[Encoded]]]:
            for index, request in enumerate(requests):
                yield index, encoder.plan(request, max_length)

        source = _prefetch(plans()) if self.options.prefetch and len(requests) > 1 else plans()
        pending: dict[int, list[Encoded]] = {}
        results: dict[int, dict[str, list[float]]] = {}
        remaining: dict[int, int] = {}
        queue_: list[tuple[int, Encoded]] = []
        queued_tokens = 0
        next_out = 0

        def flush(force: bool) -> None:
            nonlocal queue_, queued_tokens
            while queue_ and (force or queued_tokens >= budget):
                pack, used = [], 0
                for item in queue_:
                    if pack and used + item[1].length > budget:
                        break
                    pack.append(item)
                    used += item[1].length
                for (index, chunk), rows in zip(pack, self.forward([c for _, c in pack]), strict=True):
                    results.setdefault(index, {}).update(zip(chunk.keys, rows, strict=True))
                    remaining[index] -= 1
                queue_ = queue_[len(pack):]
                queued_tokens -= used
                if not force and queued_tokens < budget:
                    break

        for index, plan in source:
            pending[index] = plan
            remaining[index] = len(plan)
            for chunk in plan:
                queue_.append((index, chunk))
                queued_tokens += chunk.length
            flush(force=False)
            while next_out in pending and remaining[next_out] == 0:
                yield requests[next_out], results.pop(next_out), pending.pop(next_out)
                next_out += 1
        flush(force=True)
        while next_out in pending:
            yield requests[next_out], results.pop(next_out), pending.pop(next_out)
            next_out += 1

    # -------------------------------------------------------------- forward

    def _positions(self, chunk: Encoded) -> torch.Tensor:
        if chunk.image_grid_thw is None:
            return torch.arange(chunk.length).expand(3, -1)
        positions, _ = self.decider.backbone.get_rope_index(
            chunk.input_ids[None], chunk.mm_token_type_ids[None], image_grid_thw=chunk.image_grid_thw)
        return positions[:, 0]

    @torch.inference_mode()
    def forward(self, chunks: list[Encoded]) -> list[list[list[float]]]:
        device = self.decider.device
        lengths = [chunk.length for chunk in chunks]
        total = sum(lengths)
        bucket = next((b for b in self.buckets if b >= total), None)
        ids = [chunk.input_ids for chunk in chunks]
        positions = [self._positions(chunk) for chunk in chunks]
        if bucket is not None:  # filler sequences up to the static size; never read out
            fill = bucket - total
            while fill > 0:
                size = min(fill, FILLER_MAX)
                lengths.append(size)
                ids.append(torch.full((size,), self.decider.encoder.pad_id, dtype=torch.long))
                positions.append(torch.arange(size).expand(3, -1))
                fill -= size
        cu_cpu = F.pad(torch.tensor(lengths).cumsum(0), (1, 0))
        seqpos = torch.cat([torch.arange(n) for n in lengths]).to(device, non_blocking=True)
        input_ids = torch.cat(ids).to(device, non_blocking=True)
        rope = torch.cat(positions, dim=1)[:, None, :].to(device, non_blocking=True)
        cu = cu_cpu.to(device, torch.int32, non_blocking=True)
        _PACK.update(cu=cu, cu_long=cu.long(), cu_cpu=cu_cpu, max=max(lengths))

        h = self.lm.embed_tokens(input_ids)[None]
        pixels = [chunk.pixel_values for chunk in chunks if chunk.pixel_values is not None]
        if pixels:
            grids = torch.cat([c.image_grid_thw for c in chunks if c.image_grid_thw is not None]).to(device)
            features = self.decider.backbone.get_image_features(torch.cat(pixels).to(device), grids,
                                                                return_dict=True).pooler_output
            features = torch.cat(features).to(h.dtype)
            mask = (input_ids == self.image_token_id)[None, :, None].expand_as(h)
            h = h.masked_scatter(mask, features)
        cos, sin = self.lm.rotary_emb(h, rope)

        starts = cu_cpu[: len(chunks)].tolist()
        sel = torch.cat([chunk.markers + start for chunk, start in zip(chunks, starts, strict=True)]).to(device)
        static = bucket is not None and self.graph_fn is not None
        # A chunk longer than the budget (max_length > budget) runs alone and uncompiled.
        fn = self.graph_fn if static else self.layer_fn if total <= self.options.budget() else decoder_layer
        layers = self.layers[:-1] if self.options.prune_last else self.layers
        if static:
            torch.compiler.cudagraph_mark_step_begin()
        for layer in layers:
            h = fn(layer, h, cos, sin, seqpos, self.conv)
            if static:
                h = h.clone()  # CUDA-graph outputs are overwritten by the next replay
        if self.options.prune_last:
            per_seq = torch.tensor([0] + [len(c.keys) for c in chunks]).cumsum(0)
            cu_q = per_seq.to(device, torch.int32)
            h = self.last_fn(self.layers[-1], h, cos, sin, sel, cu_q, max(len(c.keys) for c in chunks),
                             cu[: len(chunks) + 1], max(c.length for c in chunks))
        else:
            h = h[:, sel]
        logits = self.decider.readout(self.lm.norm(h)[0]).float().cpu()
        out, cursor = [], 0
        for chunk in chunks:
            out.append([logits[cursor + j, :count].tolist() for j, count in enumerate(chunk.counts)])
            cursor += len(chunk.keys)
        return out

    def warmup(self) -> None:
        """Run one dummy pack per static bucket (or two sizes when dynamic) so compilation happens up front."""
        request = {"state": "warmup", "questions": {"q": {"type": "noul", "instructions": "Warm up?"}}}
        chunk = self.decider.encoder.plan(request, None)[0]
        sizes = self.buckets or (512, 4096)
        for size in sizes:
            count = max(1, min(size // chunk.length, 64)) if not self.buckets else 1
            for _ in range(3 if self.buckets else 2):
                self.forward([chunk] * count)
        torch.cuda.synchronize()


def _prefetch(source: Iterator[Any], depth: int = 64) -> Iterator[Any]:
    """Run ``source`` (tokenization and image preprocessing) in a background thread; re-raise its errors."""
    items: queue.Queue[Any] = queue.Queue(maxsize=depth)
    done = object()

    def produce() -> None:
        try:
            for item in source:
                items.put(item)
        except BaseException as error:  # forwarded to the consumer
            items.put(error)
        items.put(done)

    threading.Thread(target=produce, daemon=True).start()
    while (item := items.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item
