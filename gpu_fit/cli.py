"""Command-line interface for gpu-fit."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from . import __version__, core

console = Console()


def _fmt_cost(c: float | None) -> str:
    return "—  (local)" if c is None else f"${c:,.2f}"


def _fmt_hr(h: float | None) -> str:
    return "—" if h is None else f"${h:.2f}/hr"


def cmd_recommend(args: argparse.Namespace) -> int:
    model = core.find_model(args.model)
    if model is None:
        console.print(f"[red]Unknown model:[/red] {args.model}")
        console.print("Run [bold]gpu-fit models[/bold] to see what's available, "
                      "or add it to models.yaml.")
        return 2

    fits = core.recommend(
        model, ctx=args.ctx, batch=args.batch, quant=args.quant,
        budget_per_hr=args.budget_hr, allow_tp=not args.single_gpu, rank=args.rank,
    )

    header = (f"[bold]{model.name}[/bold]  ·  ctx {args.ctx:,}  ·  batch {args.batch}"
              f"  ·  quant {args.quant}  ·  ranked by {args.rank}")
    if args.budget_hr:
        header += f"  ·  budget ≤ ${args.budget_hr:.2f}/hr"
    console.print(Panel(header, expand=False))

    if not fits:
        console.print("[yellow]Nothing fits under those constraints.[/yellow] "
                      "Try a smaller ctx, allow quantization (--quant auto), "
                      "or raise --budget-hr.")
        return 1

    table = Table(show_lines=False, header_style="bold cyan")
    table.add_column("GPU")
    table.add_column("Quant")
    table.add_column("GPUs", justify="right")
    table.add_column("VRAM used", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("$/hr", justify="right")
    table.add_column("$/1M tok", justify="right")

    for i, f in enumerate(fits):
        cap = core.usable_vram(f.gpu) * f.tp
        vram_str = f"{f.vram['total']:.1f} / {cap:.0f} GiB"
        style = "bold green" if i == 0 else ""
        table.add_row(
            f.gpu.name, f.quant, str(f.tp), vram_str,
            f"{f.tok_s:,.0f}", _fmt_hr(f.usd_per_hr), _fmt_cost(f.cost_mtok),
            style=style,
        )
    console.print(table)

    top = fits[0]
    console.print()
    console.print(Panel.fit(
        core.vllm_command(model, top, args.ctx),
        title=f"[green]▶ recommended: {top.gpu.name} ({top.quant})[/green]",
        border_style="green",
    ))
    b = top.vram
    console.print(
        f"[dim]breakdown: weights {b['weights']:.1f} + KV {b['kv_cache']:.1f} "
        f"+ overhead {b['overhead']:.1f} = {b['total']:.1f} GiB "
        f"· MBU {core.MBU:.0%} · estimates, verify on your stack[/dim]"
    )
    return 0


def cmd_gpu(args: argparse.Namespace) -> int:
    model = core.find_model(args.model)
    gpu = core.find_gpu(args.gpu)
    if model is None:
        console.print(f"[red]Unknown model:[/red] {args.model}"); return 2
    if gpu is None:
        console.print(f"[red]Unknown GPU:[/red] {args.gpu}"); return 2

    f = core.best_fit_for_gpu(gpu, model, args.ctx, args.batch, args.quant,
                              allow_tp=not args.single_gpu)
    title = f"{model.name} on {gpu.name}"
    if f is None:
        console.print(Panel(
            f"[red]Does not fit[/red] even at int4"
            + ("" if args.single_gpu else " across up to 8 GPUs")
            + f".\nModel needs more memory than {gpu.name} provides at ctx {args.ctx:,}.",
            title=title, border_style="red"))
        return 1

    cap = core.usable_vram(gpu) * f.tp
    lines = [
        f"quant        {f.quant}",
        f"tensor-parallel  {f.tp}× {gpu.name}",
        f"VRAM         {f.vram['total']:.1f} / {cap:.0f} GiB usable",
        f"  weights    {f.vram['weights']:.1f} GiB",
        f"  KV cache   {f.vram['kv_cache']:.1f} GiB  (ctx {args.ctx:,} × batch {args.batch})",
        f"  overhead   {f.vram['overhead']:.1f} GiB",
        f"decode       {f.tok_s:,.0f} tok/s  (single stream, MBU {core.MBU:.0%})",
        f"price        {_fmt_hr(f.usd_per_hr)}",
        f"cost         {_fmt_cost(f.cost_mtok)} / 1M tokens",
    ]
    console.print(Panel("\n".join(lines), title=f"[green]{title}[/green]",
                        border_style="green", expand=False))
    console.print(Panel.fit(core.vllm_command(model, f, args.ctx),
                            title="serve", border_style="cyan"))
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    t = Table(header_style="bold cyan", title="Known models")
    t.add_column("Name"); t.add_column("Params", justify="right")
    t.add_column("Layers", justify="right"); t.add_column("KV heads", justify="right")
    t.add_column("HF id", style="dim")
    for m in core.load_models():
        t.add_row(m.name, f"{m.params_b:.1f}B", str(m.n_layers),
                  str(m.n_kv_heads), m.hf_id)
    console.print(t)
    return 0


def cmd_gpus(_args: argparse.Namespace) -> int:
    t = Table(header_style="bold cyan", title="Known GPUs")
    t.add_column("Name"); t.add_column("VRAM", justify="right")
    t.add_column("BW GB/s", justify="right"); t.add_column("$/hr", justify="right")
    t.add_column("note", style="dim")
    for g in core.load_gpus():
        t.add_row(g.name, f"{g.vram_gb:.0f} GiB", f"{g.mem_bw_gbps:,.0f}",
                  _fmt_hr(g.usd_per_hr), g.note)
    console.print(t)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-fit",
        description="Which GPU, quantization, and serving flags to run any LLM.")
    p.add_argument("--version", action="version", version=f"gpu-fit {__version__}")
    sub = p.add_subparsers(dest="cmd")

    def add_common(sp):
        sp.add_argument("--ctx", type=int, default=8192, help="context length (default 8192)")
        sp.add_argument("--batch", type=int, default=1, help="concurrent sequences (default 1)")
        sp.add_argument("--quant", default="auto",
                        choices=["auto", "fp16", "int8", "int4"],
                        help="quantization; 'auto' picks the best that fits")
        sp.add_argument("--single-gpu", action="store_true",
                        help="disallow multi-GPU tensor parallelism")

    rec = sub.add_parser("recommend", help="rank all GPUs for a model (default command)")
    rec.add_argument("model", help="e.g. llama-3.1-8b, qwen2.5-32b")
    rec.add_argument("--budget-hr", type=float, help="max $/hr to consider")
    rec.add_argument("--rank", choices=["cost", "speed"], default="cost")
    add_common(rec)
    rec.set_defaults(func=cmd_recommend)

    gp = sub.add_parser("gpu", help="detailed fit for one model on one GPU")
    gp.add_argument("model")
    gp.add_argument("gpu", help="e.g. 'a100 80gb', '4090'")
    add_common(gp)
    gp.set_defaults(func=cmd_gpu)

    sub.add_parser("models", help="list known models").set_defaults(func=cmd_models)
    sub.add_parser("gpus", help="list known GPUs").set_defaults(func=cmd_gpus)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Convenience: `gpu-fit llama-3.1-8b` implies the recommend subcommand.
    known = {"recommend", "gpu", "models", "gpus", "-h", "--help", "--version"}
    if argv and argv[0] not in known:
        argv = ["recommend"] + argv

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
