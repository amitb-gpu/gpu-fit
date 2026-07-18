<h1 align="center">gpu-fit</h1>

<p align="center">
  <b>Which GPU, quantization, and serving flags to run any LLM — in one command.</b><br>
  Backed by a first-principles memory + bandwidth model, not vibes.
</p>

<p align="center">
  <code>pip install gpu-fit</code> &nbsp;·&nbsp; <code>gpu-fit llama-3.1-8b</code>
</p>

---

## The problem

You want to serve an open model. Before you can even start you have to answer:

- Will it **fit**? On which card, at what context length?
- Do I need to **quantize**? int8 or int4?
- One GPU or **tensor-parallel** across several?
- What does it actually **cost per million tokens**, and where is it cheapest?
- What's the exact **`vllm serve` command**?

Today you answer these by renting a machine, OOM-ing, googling, and guessing. `gpu-fit` answers them in a second, offline, from the model architecture and GPU specs — and hands you a runnable command.

## Quickstart

```console
$ pip install gpu-fit
$ gpu-fit llama-3.1-8b
```

```
╭─ Llama-3.1-8B · ctx 8,192 · batch 1 · quant auto · ranked by cost ─╮
╰────────────────────────────────────────────────────────────────────╯
 GPU            Quant  GPUs   VRAM used        tok/s     $/hr   $/1M tok
 RTX 3090       fp16   1      17.5 / 22 GiB       41    $0.22      $1.50
 RTX 4090       fp16   1      17.5 / 22 GiB       44    $0.44      $2.78
 T4             int8   1      10.0 / 15 GiB       28    $0.35      $3.49
 H200 141GB     fp16   1      17.5 / 130 GiB     209    $4.20      $5.58
 A100 80GB      fp16   1      17.5 / 74 GiB       89    $1.80      $5.63
 A100 40GB      fp16   1      17.5 / 37 GiB       68    $1.40      $5.74
 H100 80GB SXM  fp16   1      17.5 / 74 GiB      146    $3.20      $6.09
 A10G           fp16   1      17.5 / 22 GiB       26    $0.75      $7.97
 L4             fp16   1      17.5 / 22 GiB       13    $0.70     $14.87
 RTX A1000 6GB  int4   2       6.2 / 11 GiB       67       —    — (local)

╭─ ▶ recommended: RTX 3090 (fp16) ────────────────────────────────────╮
│ vllm serve meta-llama/Llama-3.1-8B-Instruct \                        │
│     --max-model-len 8192 \                                           │
│     --gpu-memory-utilization 0.92                                    │
╰──────────────────────────────────────────────────────────────────────╯
breakdown: weights 15.0 + KV 1.0 + overhead 1.5 = 17.5 GiB · MBU 70% · estimates, verify on your stack
```

### More

```console
# Big model — see how it shards
$ gpu-fit llama-3.1-70b --rank speed

# Only what I can afford
$ gpu-fit qwen2.5-32b --budget-hr 1.50

# One model on one card, in detail
$ gpu-fit mistral-7b "a100 80gb" --ctx 32768

# Force a quantization / a single GPU
$ gpu-fit qwen2.5-72b --quant int4 --single-gpu

$ gpu-fit models      # list known models
$ gpu-fit gpus        # list known GPUs + prices
```

## How the estimate works

It's a deliberately simple, auditable roofline — every number is one you can check by hand:

**Memory** = weights + KV cache + overhead
- `weights   = params × bytes/param` (fp16=2, int8=1, int4=0.5)
- `KV cache  = 2 × layers × ctx × kv_heads × head_dim × batch × 2B` (GQA-aware)
- `overhead  ≈ 1.5 GiB` CUDA context + activations

**Decode speed** is memory-bandwidth bound — every generated token reads the full weight set once:

```
tokens/sec ≈ MBU × mem_bandwidth / weight_bytes
```

where **MBU** (memory-bandwidth utilization, default `0.70`) is the fraction of peak a real serving stack hits. **Cost/1M tokens** falls out of that throughput and the GPU's hourly price.

These are estimates that get you to the *right* GPU and config without renting ten machines. They are not a promise of exact numbers on your stack — so everything is tunable:

- `GPU_FIT_MBU=0.8 gpu-fit …` — dial utilization to match your measured value
- Edit **`gpu_fit/data/gpus.yaml`** — prices move; put in the cloud you actually rent from
- Edit **`gpu_fit/data/models.yaml`** — add any architecture; it works immediately

> The default RTX A1000 row is a measured reference point from
> [ampere-llm-perf-lab](https://github.com/amitb-gpu/ampere-llm-perf-lab). The roadmap is to
> replace *every* MBU constant with per-GPU measured values from that lab, so the estimates
> become measurements.

## Install

```console
pip install gpu-fit          # or: pipx install gpu-fit
```

From source:

```console
git clone https://github.com/amitb-gpu/gpu-fit && cd gpu-fit
pip install -e .
pytest
```

> On WSL, clone to a native Linux path (e.g. `~/`), not a Windows drive under
> `/mnt/c`: editable installs need hardlink/chmod ops that the drvfs mount blocks.

Python 3.9+. Two small deps (`pyyaml`, `rich`). No GPU or network required to run it.

## Roadmap

- [ ] Per-GPU measured MBU from `ampere-llm-perf-lab` (turn estimates into measurements)
- [ ] Prefill/TTFT modeling (compute-bound) alongside decode
- [ ] Live cloud price fetch (RunPod / Lambda / Vast) behind a flag
- [ ] `--serve llama.cpp | tgi | sglang` command emitters
- [ ] MoE models (active vs total params)
- [ ] A tiny web UI

## Sibling projects

Part of the **amitb-gpu** GPU-infra toolkit:
[ampere-llm-perf-lab](https://github.com/amitb-gpu/ampere-llm-perf-lab) ·
[gpu-rdma-roce-lab](https://github.com/amitb-gpu/gpu-rdma-roce-lab) ·
[robofleet-nexus](https://github.com/amitb-gpu/robofleet-nexus)

## License

MIT — see [LICENSE](LICENSE). PRs adding GPUs, models, and measured MBU values are very welcome.
