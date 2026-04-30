from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiment import load_wikitext, markdown_table, synthetic_data, write_plotly_html
from model import (
    TraditionalTransformerLM,
    count_parameters,
    lm_loss_from_hidden,
    lm_metrics_from_hidden,
    traditional_arch_for_budget,
    traditional_width_for_budget,
)


@dataclass(frozen=True)
class BaselineConfig:
    target_params: int
    epochs: int = 2
    seq_len: int = 128
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.1
    vocab_size: int = 32_000
    n_layers: int = 0
    n_heads: int = 0
    max_train_batches: int | None = 128
    max_eval_batches: int | None = None
    grad_accum_steps: int = 1
    compile: bool = False
    num_workers: int = 0


def result_path(cfg: BaselineConfig, out_dir: Path) -> Path:
    return out_dir / f"result_traditional_p{cfg.target_params}.json"


@torch.no_grad()
def evaluate_baseline(
    model: TraditionalTransformerLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        metrics = lm_metrics_from_hidden(
            model,
            x.to(device, non_blocking=device.type == "cuda"),
            y.to(device, non_blocking=device.type == "cuda"),
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    return {key: value / max(1, n) for key, value in totals.items()}


def run_baseline(
    cfg: BaselineConfig,
    out_dir: Path,
    datasets: tuple[TensorDataset, TensorDataset],
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_ds, val_ds = datasets
    if cfg.n_layers > 0 and cfg.n_heads > 0:
        n_layers = cfg.n_layers
        n_heads = cfg.n_heads
        d_model = traditional_width_for_budget(
            cfg.target_params,
            cfg.vocab_size,
            cfg.seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        architecture_strategy = "fixed_depth_width_search"
    else:
        d_model, n_layers, n_heads = traditional_arch_for_budget(
            cfg.target_params,
            cfg.vocab_size,
            cfg.seq_len,
        )
        architecture_strategy = "auto_gpt_style_depth_width_search"
    model = TraditionalTransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=cfg.seq_len,
    ).to(device)
    actual_params = count_parameters(model)
    logit_scale = model.logit_scale
    if cfg.compile:
        model = torch.compile(model)

    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if cfg.batch_size % cfg.grad_accum_steps != 0:
        raise ValueError("batch_size must be divisible by grad_accum_steps")

    micro_batch_size = cfg.batch_size // cfg.grad_accum_steps
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"
    opt_kwargs = {"fused": True} if device.type == "cuda" else {}
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, **opt_kwargs
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(micro_batch_size, 16),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    curve_path = out_dir / f"curve_traditional_p{cfg.target_params}.csv"

    rows = []
    for epoch in range(cfg.epochs):
        model.train()
        micro_iterator = iter(train_loader)
        updates_per_epoch = cfg.max_train_batches or len(train_loader) // cfg.grad_accum_steps
        iterator = tqdm(
            range(updates_per_epoch),
            desc=f"traditional p={cfg.target_params} e={epoch}",
            mininterval=10,
        )
        for step in iterator:
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                try:
                    x, y = next(micro_iterator)
                except StopIteration:
                    micro_iterator = iter(train_loader)
                    x, y = next(micro_iterator)
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = lm_loss_from_hidden(model, x, y)
                    scaled_loss = loss / cfg.grad_accum_steps
                scaled_loss.backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss = total_loss / cfg.grad_accum_steps
            rows.append({"epoch": epoch, "step": step, "train_loss": train_loss})
            iterator.set_postfix(loss=f"{train_loss:.3f}")

    metrics = evaluate_baseline(model, val_loader, device, cfg.max_eval_batches)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)

    eval_sequences_seen = (
        len(val_loader.dataset)
        if cfg.max_eval_batches is None
        else cfg.max_eval_batches * min(micro_batch_size, 16)
    )
    return {
        **asdict(cfg),
        "model_type": "traditional_transformer_lm",
        "architecture_strategy": architecture_strategy,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "logit_scale": logit_scale,
        "actual_trainable_params": actual_params,
        "effective_train_batch_size": cfg.batch_size,
        "micro_train_batch_size": micro_batch_size,
        "effective_eval_batch_size": min(micro_batch_size, 16),
        "train_sequences_seen": cfg.epochs
        * (cfg.max_train_batches or len(train_loader))
        * cfg.batch_size,
        "train_tokens_seen": cfg.epochs
        * (cfg.max_train_batches or len(train_loader))
        * cfg.batch_size
        * cfg.seq_len,
        "eval_sequences_seen": eval_sequences_seen,
        "eval_tokens_seen": eval_sequences_seen * cfg.seq_len,
        **metrics,
    }


def plot_baselines(results: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    curve_paths = sorted(out_dir.glob("curve_traditional_p*.csv"))
    traces = []
    plt.figure(figsize=(8, 5))
    for path in curve_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.reset_index(names="update")
        label = path.stem.removeprefix("curve_traditional_")
        plt.plot(df["update"], df["train_loss"], label=label)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": label,
                "x": df["update"].tolist(),
                "y": df["train_loss"].tolist(),
            }
        )
    plt.xlabel("Optimizer update")
    plt.ylabel("Training loss")
    plt.title("Traditional LM Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves_traditional.png", dpi=160)
    plt.close()
    write_plotly_html(
        out_dir / "training_curves_traditional.html",
        "Traditional LM Training Curves",
        traces,
        {
            "xaxis": {"title": "Optimizer update"},
            "yaxis": {"title": "Training loss"},
            "hovermode": "x unified",
        },
    )

    df = pd.DataFrame(results).sort_values("target_params")
    val_traces = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "traditional LM",
            "x": df["target_params"].tolist(),
            "y": df["validation_loss"].tolist(),
        }
    ]
    plt.figure(figsize=(7, 4))
    plt.plot(df["target_params"], df["validation_loss"], marker="o")
    plt.xscale("log")
    plt.xlabel("Target trainable params")
    plt.ylabel("Final held-out validation loss")
    plt.tight_layout()
    plt.savefig(out_dir / "validation_loss_vs_params_traditional.png", dpi=160)
    plt.close()
    write_plotly_html(
        out_dir / "validation_loss_vs_params_traditional.html",
        "Traditional LM Validation Loss vs Params",
        val_traces,
        {
            "xaxis": {"title": "Target trainable params", "type": "log"},
            "yaxis": {"title": "Final held-out validation loss"},
            "hovermode": "x unified",
        },
    )


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(results).sort_values("target_params")
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    if df.empty:
        return
    plot_baselines(results, out_dir)

    table_cols = [
        "target_params",
        "actual_trainable_params",
        "n_layers",
        "n_heads",
        "d_model",
        "validation_loss",
        "certainty_mean",
        "train_tokens_seen",
        "eval_tokens_seen",
    ]
    table = df[table_cols].copy()
    for col in table.select_dtypes(include="number").columns:
        if col in {"validation_loss", "certainty_mean"}:
            table[col] = table[col].map(lambda value: f"{float(value):.6f}")
        else:
            table[col] = table[col].map(lambda value: f"{int(value)}")

    lines = [
        "# Traditional LM Baseline",
        "",
        "## What Was Run",
        "",
        "This trains a single conventional GPT-style causal Transformer language model at the requested trainable-parameter targets. It uses the same Wikitext-103 BPE cache, vocabulary size 32,000, sequence length 128, AdamW settings, 256 optimizer updates, and 393,216 training tokens per run as the all-columns Latin-square run.",
        "",
        "The model uses tied input/output token embeddings, learned positional embeddings, pre-LN causal self-attention blocks, GPT-style normal initialization, residual projection scaling, and logits scaled by `1/sqrt(d_model)`. Unless fixed depth/head settings are supplied, the runner searches over conventional depth, width, and head-count choices and selects the largest stable GPT-style architecture under the target parameter budget.",
        "",
        "## Results",
        "",
        markdown_table(table),
        "",
        "## Plots",
        "",
        "Interactive training curves are in `training_curves_traditional.html`. Interactive validation-loss comparison by parameter budget is in `validation_loss_vs_params_traditional.html`.",
        "",
        "## Comparison Note",
        "",
        "These baselines are parameter-capped including tied embeddings and all Transformer weights. For parameter-matched reruns, the target parameter counts should be the Latin-square rows' actual trainable-parameter counts, not the nominal submodel-budget labels.",
    ]
    (out_dir / "WRITEUP.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/traditional_baseline"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-train-batches", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=0)
    parser.add_argument("--n-heads", type=int, default=0)
    parser.add_argument(
        "--target-params",
        type=str,
        help="comma-separated target trainable parameter counts",
    )
    parser.add_argument(
        "--target-updates",
        type=str,
        help="comma-separated optimizer-update counts matching --target-params",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.target_params:
        target_params = [int(value) for value in args.target_params.split(",")]
    elif args.smoke:
        target_params = [1_000_000]
    else:
        target_params = [1_000_000, 10_000_000]
    target_updates = None
    if args.target_updates:
        target_updates = [int(value) for value in args.target_updates.split(",")]
        if len(target_updates) != len(target_params):
            raise ValueError("--target-updates must have the same length as --target-params")
    configs = [
        BaselineConfig(
            target_params=target,
            epochs=1 if args.smoke or target_updates else args.epochs,
            seq_len=min(args.seq_len, 128) if args.smoke else args.seq_len,
            batch_size=min(args.batch_size, 8) if args.smoke else args.batch_size,
            max_train_batches=1
            if args.smoke
            else target_updates[i]
            if target_updates
            else args.max_train_batches,
            max_eval_batches=1 if args.smoke else args.max_eval_batches,
            grad_accum_steps=args.grad_accum_steps,
            compile=args.compile,
            num_workers=args.num_workers,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
        for i, target in enumerate(target_params)
    ]
    datasets = (
        synthetic_data(configs[0].vocab_size, configs[0].seq_len)
        if args.smoke
        else load_wikitext(configs[0].vocab_size, configs[0].seq_len)
    )

    results = []
    for cfg in configs:
        path = result_path(cfg, args.out_dir)
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        result = run_baseline(cfg, args.out_dir, datasets)
        path.write_text(json.dumps(result, indent=2))
        results.append(result)
        write_summary(results, args.out_dir)
    write_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
