"""Core memory-fit, throughput, and cost model for gpu-fit.

The estimates here are first-principles roofline approximations, deliberately
simple and auditable:

* VRAM  = weights + KV cache + framework/activation overhead
* Decode throughput (tokens/s, single stream) is memory-bandwidth bound:
  each generated token reads the full weight set once, so
      tok/s ~= MBU * mem_bandwidth / weight_bytes
  where MBU (memory-bandwidth utilization) is the fraction of peak actually
  achieved by a real serving stack. Default 0.70 is a conservative,
  measured-in-practice value; override with GPU_FIT_MBU.
* Cost per 1M tokens is derived from the GPU's hourly price and that throughput.

These are estimates, not guarantees. They get you to the right GPU and config
without renting ten machines first. Replace MBU / prices / arch rows with your
own measurements to make them exact for your stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # py3.9+ stdlib
    from importlib.resources import files as _res_files
except ImportError:  # pragma: no cover
    from importlib_resources import files as _res_files  # type: ignore

GIB = 1024 ** 3
# Fraction of peak memory bandwidth a real serving stack achieves during decode.
MBU = float(os.environ.get("GPU_FIT_MBU", "0.70"))
# Framework/CUDA-context + activation overhead reserved per GPU, in GiB.
OVERHEAD_GIB = float(os.environ.get("GPU_FIT_OVERHEAD_GIB", "1.5"))
# Usable fraction of advertised VRAM (fragmentation, allocator slack).
VRAM_USABLE = 0.92

BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
# Quantization quality order, best first.
QUANT_ORDER = ["fp16", "int8", "int4"]


@dataclass
class GPU:
    name: str
    vram_gb: float
    mem_bw_gbps: float
    fp16_tflops: float = 0.0
    usd_per_hr: float | None = None
    local: bool = False
    note: str = ""


@dataclass
class Model:
    name: str
    params_b: float
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hf_id: str = ""


def _load(fname: str) -> list[dict]:
    import yaml  # imported lazily: only needed to read the bundled data files
    text = _res_files("gpu_fit.data").joinpath(fname).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_gpus() -> list[GPU]:
    return [GPU(**row) for row in _load("gpus.yaml")]


def load_models() -> list[Model]:
    return [Model(**row) for row in _load("models.yaml")]


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def find_model(query: str, models: list[Model] | None = None) -> Model | None:
    """Loose, case-insensitive match: 'llama3.1-8b', 'llama31 8b', 'Llama-3.1-8B'."""
    models = models or load_models()
    q = _norm(query)
    for m in models:
        if _norm(m.name) == q:
            return m
    # substring / alias fallback
    for m in models:
        if q and (q in _norm(m.name) or _norm(m.name) in q):
            return m
    return None


def find_gpu(query: str, gpus: list[GPU] | None = None) -> GPU | None:
    gpus = gpus or load_gpus()
    q = _norm(query)
    for g in gpus:
        if _norm(g.name) == q:
            return g
    for g in gpus:
        if q and q in _norm(g.name):
            return g
    return None


# --- memory model -----------------------------------------------------------

def weight_gib(model: Model, quant: str) -> float:
    return model.params_b * 1e9 * BYTES_PER_PARAM[quant] / GIB


def kv_cache_gib(model: Model, ctx: int, batch: int, kv_bytes: float = 2.0) -> float:
    """KV cache for `batch` concurrent sequences of length `ctx`.

    2 (K+V) * layers * ctx * (kv_heads * head_dim) * batch * bytes_per_element.
    """
    per_token = 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_bytes
    return per_token * ctx * batch / GIB


def total_vram_gib(model: Model, quant: str, ctx: int, batch: int) -> dict:
    w = weight_gib(model, quant)
    kv = kv_cache_gib(model, ctx, batch)
    total = w + kv + OVERHEAD_GIB
    return {"weights": w, "kv_cache": kv, "overhead": OVERHEAD_GIB, "total": total}


def usable_vram(gpu: GPU) -> float:
    return gpu.vram_gb * VRAM_USABLE


def fits_on(gpu: GPU, model: Model, quant: str, ctx: int, batch: int, tp: int) -> bool:
    """Does `model` fit across `tp` copies of `gpu`? Weights and KV shard by tp;
    overhead is charged per-GPU."""
    w = weight_gib(model, quant)
    kv = kv_cache_gib(model, ctx, batch)
    return (w + kv) / tp + OVERHEAD_GIB <= usable_vram(gpu)


# --- performance + cost -----------------------------------------------------

def decode_tok_s(gpu: GPU, model: Model, quant: str, tp: int = 1) -> float:
    """Single-stream decode tokens/sec (memory-bandwidth roofline).

    Aggregate server throughput scales roughly with batch size on top of this;
    this figure is the per-request speed a user feels.
    """
    w_bytes = model.params_b * 1e9 * BYTES_PER_PARAM[quant]
    eff_bw = gpu.mem_bw_gbps * 1e9 * tp  # TP adds bandwidth
    return MBU * eff_bw / w_bytes


def cost_per_mtok(gpu: GPU, tok_s: float, tp: int = 1) -> float | None:
    if gpu.usd_per_hr is None or tok_s <= 0:
        return None
    usd_per_sec = gpu.usd_per_hr * tp / 3600.0
    return usd_per_sec / tok_s * 1e6


# --- recommendation ---------------------------------------------------------

@dataclass
class Fit:
    gpu: GPU
    quant: str
    tp: int
    vram: dict
    tok_s: float
    cost_mtok: float | None

    @property
    def usd_per_hr(self) -> float | None:
        return None if self.gpu.usd_per_hr is None else self.gpu.usd_per_hr * self.tp


def best_fit_for_gpu(gpu: GPU, model: Model, ctx: int, batch: int,
                     quant: str = "auto", allow_tp: bool = True) -> Fit | None:
    """Cheapest sensible fit: fewest GPUs first, then highest quality.

    We'd rather run int8 on one card than fp16 on two, so tensor-parallelism is
    only introduced when a single card can't hold the model even at int4. Force a
    specific quant (or single GPU) to override.
    """
    quants = QUANT_ORDER if quant == "auto" else [quant]
    tps = [n for n in (1, 2, 4, 8) if n <= (8 if allow_tp else 1)]
    for tp in tps:
        for q in quants:  # within a GPU count, prefer the highest quality
            if fits_on(gpu, model, q, ctx, batch, tp):
                vram = total_vram_gib(model, q, ctx, batch)
                tok_s = decode_tok_s(gpu, model, q, tp)
                return Fit(gpu, q, tp, vram, tok_s, cost_per_mtok(gpu, tok_s, tp))
    return None


def recommend(model: Model, ctx: int = 8192, batch: int = 1, quant: str = "auto",
              budget_per_hr: float | None = None, allow_tp: bool = True,
              gpus: list[GPU] | None = None, rank: str = "cost") -> list[Fit]:
    """Rank every GPU that can serve `model`.

    rank: 'cost' (cheapest $/Mtok first) or 'speed' (fastest tok/s first).
    Local (unpriced) cards are always kept and sorted last within their bucket.
    """
    gpus = gpus or load_gpus()
    fits: list[Fit] = []
    for g in gpus:
        f = best_fit_for_gpu(g, model, ctx, batch, quant, allow_tp)
        if f is None:
            continue
        if budget_per_hr is not None and f.usd_per_hr is not None and f.usd_per_hr > budget_per_hr:
            continue
        fits.append(f)

    def key(f: Fit):
        if rank == "speed":
            return (-f.tok_s,)
        # cost: priced first (cheapest), then local/unpriced by speed
        if f.cost_mtok is None:
            return (1, -f.tok_s)
        return (0, f.cost_mtok)

    return sorted(fits, key=key)


def vllm_command(model: Model, fit: Fit, ctx: int) -> str:
    parts = [f"vllm serve {model.hf_id or model.name}",
             f"--max-model-len {ctx}",
             f"--gpu-memory-utilization {VRAM_USABLE}"]
    if fit.tp > 1:
        parts.append(f"--tensor-parallel-size {fit.tp}")
    if fit.quant == "int4":
        parts.append("--quantization awq  # requires an AWQ/GPTQ checkpoint")
    elif fit.quant == "int8":
        parts.append("--quantization fp8  # or load an INT8/W8A8 checkpoint")
    return " \\\n    ".join(parts)
