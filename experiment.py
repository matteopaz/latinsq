from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import (
    LatinEnsembleLM,
    ensemble_loss_from_hidden,
    ensemble_metrics_from_hidden,
    submodel_width,
)
from square import max_entropy_latin_square, min_entropy_latin_square


@dataclass(frozen=True)
class Config:
    k: int
    total_params: int
    scheduler: str
    epochs: int = 2
    seq_len: int = 128
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    vocab_size: int = 32_000
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    compile: bool = False
    num_workers: int = 0
    auto_batch_by_k: bool = False
    active_column_fraction: float = 0.25
    active_row_fraction: float = 1.0
    grad_accum_steps: int = 1


def schedule(name: str, k: int) -> list[list[int]]:
    if name == "min":
        return min_entropy_latin_square(k)
    if name == "max":
        return max_entropy_latin_square(k)
    raise ValueError(f"unknown scheduler: {name}")


def load_wikitext(
    vocab_size: int, seq_len: int, cache_dir: Path = Path("data/cache")
) -> tuple[TensorDataset, TensorDataset]:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_cache = cache_dir / f"wikitext103_bpe{vocab_size}_seq{seq_len}.pt"
    if tensor_cache.exists():
        data = torch.load(tensor_cache, weights_only=True)
        return TensorDataset(data["train_x"], data["train_y"]), TensorDataset(
            data["val_x"], data["val_y"]
        )

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    texts = (row["text"] for row in ds["train"] if row["text"].strip())
    tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(split: str) -> torch.Tensor:
        ids: list[int] = []
        eos = tokenizer.token_to_id("<eos>")
        for row in ds[split]:
            if row["text"].strip():
                ids.extend(tokenizer.encode(row["text"]).ids)
                ids.append(eos)
        usable = ((len(ids) - 1) // seq_len) * seq_len
        x = torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq_len)
        y = torch.tensor(ids[1 : usable + 1], dtype=torch.long).view(-1, seq_len)
        return TensorDataset(x, y)

    train = encode("train")
    val = encode("validation")
    torch.save(
        {
            "train_x": train.tensors[0],
            "train_y": train.tensors[1],
            "val_x": val.tensors[0],
            "val_y": val.tensors[1],
        },
        tensor_cache,
    )
    return train, val


def synthetic_data(
    vocab_size: int, seq_len: int, n: int = 128
) -> tuple[TensorDataset, TensorDataset]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab_size, (n, seq_len), generator=gen)
    y = torch.roll(x, shifts=-1, dims=1)
    return TensorDataset(x, y), TensorDataset(x[: n // 4], y[: n // 4])


def run_one(
    cfg: Config,
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
    width = submodel_width(cfg.total_params, cfg.k)
    model = LatinEnsembleLM(
        vocab_size=cfg.vocab_size,
        schedule=schedule(cfg.scheduler, cfg.k),
        d_model=width,
        max_seq_len=cfg.seq_len,
        active_column_fraction=cfg.active_column_fraction,
        active_row_fraction=cfg.active_row_fraction,
    ).to(device)
    if cfg.compile:
        model = torch.compile(model)
    use_amp = device.type == "cuda"
    opt_kwargs = {}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, **opt_kwargs
    )
    pin_memory = device.type == "cuda"
    train_batch_size = cfg.batch_size
    if cfg.auto_batch_by_k:
        active_column_count = max(1, math.ceil(cfg.k * cfg.active_column_fraction))
        train_batch_size = max(1, cfg.batch_size * 4 // active_column_count)
        train_batch_size = min(train_batch_size, cfg.batch_size)
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if train_batch_size % cfg.grad_accum_steps != 0:
        raise ValueError(
            "effective train batch size must be divisible by grad_accum_steps"
        )
    micro_train_batch_size = train_batch_size // cfg.grad_accum_steps
    eval_batch_size = min(micro_train_batch_size, 16)
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    curve_path = out_dir / f"curve_k{cfg.k}_p{cfg.total_params}_{cfg.scheduler}.csv"

    rows = []
    for epoch in range(cfg.epochs):
        model.train()
        desc = f"k={cfg.k} p={cfg.total_params} {cfg.scheduler} e={epoch}"
        micro_iterator = iter(train_loader)
        updates_per_epoch = cfg.max_train_batches or len(train_loader) // cfg.grad_accum_steps
        iterator = tqdm(range(updates_per_epoch), desc=desc, mininterval=10)
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
                    loss = ensemble_loss_from_hidden(model, x, y)
                    scaled_loss = loss / cfg.grad_accum_steps
                scaled_loss.backward()
                total_loss += loss.item()
            if cfg.max_train_batches is not None and step >= cfg.max_train_batches:
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss = total_loss / cfg.grad_accum_steps
            rows.append({"epoch": epoch, "step": step, "train_loss": train_loss})
            iterator.set_postfix(loss=f"{train_loss:.3f}")

    metrics = evaluate(model, val_loader, device, cfg.max_eval_batches)
    active_column_count = model.active_column_count()
    active_row_count = model.active_row_count()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        **asdict(cfg),
        "effective_train_batch_size": train_batch_size,
        "micro_train_batch_size": micro_train_batch_size,
        "effective_eval_batch_size": eval_batch_size,
        "active_column_fraction": cfg.active_column_fraction,
        "active_row_fraction": cfg.active_row_fraction,
        "active_column_count": active_column_count,
        "active_row_count": active_row_count,
        "train_sequences_seen": cfg.epochs
        * (cfg.max_train_batches or len(train_loader))
        * train_batch_size,
        "train_tokens_seen": cfg.epochs
        * (cfg.max_train_batches or len(train_loader))
        * train_batch_size
        * cfg.seq_len,
        "eval_sequences_seen": len(val_loader.dataset)
        if cfg.max_eval_batches is None
        else cfg.max_eval_batches * eval_batch_size,
        "eval_tokens_seen": (
            len(val_loader.dataset)
            if cfg.max_eval_batches is None
            else cfg.max_eval_batches * eval_batch_size
        )
        * cfg.seq_len,
        "d_model": width,
        **metrics,
    }


def result_path(cfg: Config, out_dir: Path) -> Path:
    return out_dir / f"result_k{cfg.k}_p{cfg.total_params}_{cfg.scheduler}.json"


@torch.no_grad()
def evaluate(
    model: LatinEnsembleLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    member_certainty_totals: Tensor | None = None
    member_certainty_counts: Tensor | None = None
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        metrics = ensemble_metrics_from_hidden(
            model,
            x.to(device, non_blocking=device.type == "cuda"),
            y.to(device, non_blocking=device.type == "cuda"),
        )
        for key, value in metrics.items():
            if key == "active_column_indices":
                continue
            if key == "member_certainty_by_index":
                active_columns = torch.tensor(metrics["active_column_indices"])
                values = torch.tensor(value)
                if member_certainty_totals is None:
                    member_certainty_totals = torch.zeros(model.k)
                    member_certainty_counts = torch.zeros(model.k)
                member_certainty_totals.index_add_(0, active_columns, values)
                member_certainty_counts.index_add_(
                    0, active_columns, torch.ones_like(values)
                )
                continue
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    averaged: dict[str, Any] = {key: value / max(1, n) for key, value in totals.items()}
    if member_certainty_totals is not None and member_certainty_counts is not None:
        observed = member_certainty_counts > 0
        member_certainty_tensor = torch.full_like(member_certainty_totals, float("nan"))
        member_certainty_tensor[observed] = (
            member_certainty_totals[observed] / member_certainty_counts[observed]
        )
        member_certainty = member_certainty_tensor.tolist()
        averaged["member_certainty_by_index"] = member_certainty
        observed_values = member_certainty_tensor[observed]
        max_value = observed_values.max().item()
        min_value = observed_values.min().item()
        averaged["most_certain_member_index"] = int(member_certainty.index(max_value))
        averaged["least_certain_member_index"] = int(member_certainty.index(min_value))
        averaged["member_certainty_range"] = float(max_value - min_value)
        averaged["member_certainty_observed_count_min"] = int(
            member_certainty_counts.min().item()
        )
        averaged["member_certainty_observed_count_max"] = int(
            member_certainty_counts.max().item()
        )
    return averaged


def plot_training_curves(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    curve_paths = sorted(out_dir.glob("curve_k*_p*_*.csv"))
    if not curve_paths:
        return

    interactive_traces = []
    plt.figure(figsize=(10, 6))
    for curve_path in curve_paths:
        df = pd.read_csv(curve_path)
        if df.empty:
            continue
        label = curve_path.stem.removeprefix("curve_").replace("_", " ")
        df = df.reset_index(names="update")
        plt.plot(df["update"], df["train_loss"], linewidth=1.2, alpha=0.85, label=label)
        interactive_traces.append(
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
    plt.title("Training curves")
    plt.legend(fontsize="small", ncols=2)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves_all.png", dpi=160)
    plt.close()

    for curve_path in curve_paths:
        df = pd.read_csv(curve_path)
        if df.empty:
            continue
        plt.figure(figsize=(7, 4))
        df = df.reset_index(names="update")
        plt.plot(df["update"], df["train_loss"], linewidth=1.5)
        plt.xlabel("Optimizer update")
        plt.ylabel("Training loss")
        plt.title(curve_path.stem.removeprefix("curve_").replace("_", " "))
        plt.tight_layout()
        plt.savefig(out_dir / f"{curve_path.stem}.png", dpi=160)
        plt.close()

    write_plotly_html(
        out_dir / "training_curves_all.html",
        "Training Curves",
        interactive_traces,
        {
            "xaxis": {"title": "Optimizer update"},
            "yaxis": {"title": "Training loss"},
            "hovermode": "x unified",
        },
    )


def write_plotly_html(
    path: Path,
    title: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
) -> None:
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = {json.dumps({"title": title, **layout})};
    Plotly.newPlot("plot", traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    path.write_text(document)


def markdown_table(df: Any) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    rows.extend([list(row) for row in df.itertuples(index=False, name=None)])
    return "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )


def write_report(results: list[dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd

    if not results:
        return
    df = pd.DataFrame(results).sort_values(["total_params", "scheduler", "k"])
    table_cols = [
        "k",
        "total_params",
        "scheduler",
        "d_model",
        "ensemble_loss",
        "avg_individual_loss",
        "ensemble_gain",
        "member_certainty_mean",
        "member_certainty_std",
        "member_certainty_range",
        "prediction_dissimilarity_kl",
    ]
    available_cols = [col for col in table_cols if col in df.columns]
    table_df = df[available_cols].copy()
    for col in table_df.select_dtypes(include="number").columns:
        if col in {"k", "total_params", "d_model"}:
            table_df[col] = table_df[col].map(lambda value: f"{int(value)}")
        else:
            table_df[col] = table_df[col].map(lambda value: f"{float(value):.6f}")
    table = markdown_table(table_df)

    scope_cols = [
        "k",
        "total_params",
        "scheduler",
        "max_train_batches",
        "max_eval_batches",
        "effective_train_batch_size",
        "micro_train_batch_size",
        "grad_accum_steps",
        "effective_eval_batch_size",
        "active_column_fraction",
        "active_row_fraction",
        "active_column_count",
        "active_row_count",
        "train_sequences_seen",
        "train_tokens_seen",
        "eval_sequences_seen",
        "eval_tokens_seen",
    ]
    scope_df = df[[col for col in scope_cols if col in df.columns]].copy()
    for col in scope_df.select_dtypes(include="number").columns:
        if col == "active_column_fraction":
            scope_df[col] = scope_df[col].map(lambda value: f"{float(value):.2f}")
        elif col == "active_row_fraction":
            scope_df[col] = scope_df[col].map(lambda value: f"{float(value):.2f}")
        else:
            scope_df[col] = scope_df[col].map(lambda value: f"{int(value)}")
    scope_table = markdown_table(scope_df)

    winner = df.loc[df["ensemble_loss"].idxmin()]
    gain_by_params = (
        df.groupby("total_params")["ensemble_gain"].mean().sort_index().to_dict()
        if "ensemble_gain" in df
        else {}
    )
    dissim_by_scheduler = (
        df.groupby("scheduler")["prediction_dissimilarity_kl"].mean().sort_index().to_dict()
        if "prediction_dissimilarity_kl" in df
        else {}
    )
    certainty_by_scheduler = (
        df.groupby("scheduler")["member_certainty_range"].mean().sort_index().to_dict()
        if "member_certainty_range" in df
        else {}
    )
    active_fractions = sorted(float(value) for value in df["active_column_fraction"].unique())
    if active_fractions == [1.0]:
        activation_text = (
            "The architecture used here activates all Latin-square columns on each "
            "forward pass. For example, `K=128` evaluates all 128 ensemble paths "
            "instead of subsampling columns."
        )
    elif len(active_fractions) == 1:
        fraction = active_fractions[0]
        example_active = int(math.ceil(128 * fraction))
        activation_text = (
            f"The architecture used here activates {fraction:.2f} of the Latin-square "
            f"columns on each forward pass. For example, `K=128` evaluates "
            f"{example_active} ensemble paths per batch instead of all 128. "
            "Inactive columns are sampled on later forwards, so the run keeps the "
            "global Latin-square structure while reducing per-step compute and memory."
        )
    else:
        activation_text = (
            "The active Latin-square column fraction varies by run; the exact active "
            "column counts are recorded in the run-scope table."
        )
    grad_accum_steps = sorted(int(value) for value in df["grad_accum_steps"].unique())
    if grad_accum_steps == [1]:
        training_text = (
            "Training uses AdamW with weight decay 0.1. The run is deliberately capped "
            "at a comparable early-convergence budget rather than full Wikitext epochs: "
            "the training curves from the previous pilot showed that most loss reduction "
            "happened by about 256 optimizer updates. The exact batch caps, effective "
            "batch sizes, and token exposure are recorded below."
        )
    else:
        training_text = (
            "Training uses AdamW with weight decay 0.1 and gradient accumulation where "
            "needed to keep the effective batch size comparable while fitting memory. "
            "The run is deliberately capped at a comparable early-convergence budget "
            "rather than full Wikitext epochs: the training curves from the previous "
            "pilot showed that most loss reduction happened by about 256 optimizer "
            "updates. The exact microbatch sizes, accumulation factors, effective "
            "batch sizes, and token exposure are recorded below."
        )

    lines = [
        "# Latin Square Ensemble Experiment",
        "",
        "## What Was Run",
        "",
        "This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.",
        "",
        activation_text,
        "",
        training_text,
        "",
        "## Run Scope",
        "",
        scope_table,
        "",
        "## Results Table",
        "",
        table,
        "",
        "## Validation Loss",
        "",
        "The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.",
        "",
        "## Training Curves",
        "",
        "The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.",
        "",
        "## Ensembling Analysis",
        "",
        f"- Best final held-out ensemble loss: {winner['ensemble_loss']:.6f} at K={int(winner['k'])}, params={int(winner['total_params'])}, scheduler={winner['scheduler']}.",
        f"- Certainty consistency: member-certainty ranges averaged by scheduler were {json.dumps(certainty_by_scheduler)}. Larger ranges indicate that some final sequence members are consistently more certain.",
        f"- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {json.dumps(gain_by_params)}. Positive values mean ensembling improved held-out CE over the average individual member.",
        f"- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {json.dumps(dissim_by_scheduler)}. Higher values indicate more diverse member predictions.",
        "",
        "## Additional Findings",
        "",
        "The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.",
        "",
        "The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.",
    ]
    (out_dir / "WRITEUP.md").write_text("\n".join(lines) + "\n")


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    interactive_traces = []
    for params in sorted(df["total_params"].unique()):
        subset = df[df["total_params"] == params]
        for scheduler in sorted(subset["scheduler"].unique()):
            trace = subset[subset["scheduler"] == scheduler].sort_values("k")
            label = f"{params} {scheduler}"
            plt.plot(trace["k"], trace["ensemble_loss"], marker="o", label=label)
            interactive_traces.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": label,
                    "x": trace["k"].tolist(),
                    "y": trace["ensemble_loss"].tolist(),
                }
            )
    plt.xlabel("K")
    plt.ylabel("Final held-out validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_loss_vs_k.png", dpi=160)
    plt.close()
    write_plotly_html(
        out_dir / "validation_loss_vs_k.html",
        "Validation Loss vs K",
        interactive_traces,
        {
            "xaxis": {"title": "K"},
            "yaxis": {"title": "Final held-out validation loss"},
            "hovermode": "x unified",
        },
    )
    plot_training_curves(out_dir)
    write_report(results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true", help="use synthetic data and one tiny run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--auto-batch-by-k",
        action="store_true",
        help="scale the effective batch size down as K grows to keep high-K runs in memory",
    )
    parser.add_argument(
        "--active-column-fraction",
        type=float,
        default=0.25,
        help="fraction of Latin-square columns active on each forward pass",
    )
    parser.add_argument(
        "--active-row-fraction",
        type=float,
        default=1.0,
        help="fraction of Latin-square rows active on each forward pass",
    )
    parser.add_argument(
        "--ks",
        type=str,
        help="comma-separated K values to run; default is 16,64,128",
    )
    parser.add_argument(
        "--total-params-list",
        type=str,
        help="comma-separated parameter budgets to run; default is 1000000,10000000",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        help="comma-separated schedulers to run; default is min,max",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="number of microbatches to accumulate per optimizer update",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        configs = [
            Config(
                k=2,
                total_params=20_000,
                scheduler="min",
                epochs=1,
                seq_len=min(args.seq_len, 128),
                batch_size=min(args.batch_size, 8),
                max_train_batches=1,
                max_eval_batches=1,
                compile=args.compile,
                num_workers=args.num_workers,
                auto_batch_by_k=args.auto_batch_by_k,
                active_column_fraction=args.active_column_fraction,
                active_row_fraction=args.active_row_fraction,
                grad_accum_steps=args.grad_accum_steps,
            )
        ]
    else:
        ks = [int(value) for value in args.ks.split(",")] if args.ks else [16, 64, 128]
        total_params_list = (
            [int(value) for value in args.total_params_list.split(",")]
            if args.total_params_list
            else [1_000_000, 10_000_000]
        )
        schedulers = args.schedulers.split(",") if args.schedulers else ["min", "max"]
        configs = [
            Config(
                k=k,
                total_params=params,
                scheduler=scheduler,
                epochs=args.epochs,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                compile=args.compile,
                num_workers=args.num_workers,
                auto_batch_by_k=args.auto_batch_by_k,
                active_column_fraction=args.active_column_fraction,
                active_row_fraction=args.active_row_fraction,
                grad_accum_steps=args.grad_accum_steps,
            )
            for k in ks
            for params in total_params_list
            for scheduler in schedulers
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
        result = run_one(cfg, args.out_dir, datasets)
        path.write_text(json.dumps(result, indent=2))
        results.append(result)
        write_summary(results, args.out_dir)
    write_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
