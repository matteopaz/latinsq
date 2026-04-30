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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiment import load_wikitext
from model import LatinEnsembleLM, count_parameters, ensemble_loss_from_hidden, submodel_width
from square import max_entropy_latin_square, min_entropy_latin_square


@dataclass(frozen=True)
class ParticipationConfig:
    k: int = 128
    total_params: int = 10_000_000
    seq_len: int = 128
    batch_size: int = 12
    grad_accum_steps: int = 12
    epochs: int = 2
    max_train_batches: int = 128
    max_eval_batches: int = 128
    vocab_size: int = 32_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    num_workers: int = 2
    subset_samples: int = 16
    seed: int = 0
    vocab_chunk_size: int = 2048


def schedule(name: str, k: int) -> list[list[int]]:
    if name == "min":
        return min_entropy_latin_square(k)
    if name == "max":
        return max_entropy_latin_square(k)
    raise ValueError(f"unknown scheduler: {name}")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: ParticipationConfig, scheduler: str) -> LatinEnsembleLM:
    return LatinEnsembleLM(
        vocab_size=cfg.vocab_size,
        schedule=schedule(scheduler, cfg.k),
        d_model=submodel_width(cfg.total_params, cfg.k),
        max_seq_len=cfg.seq_len,
        active_column_fraction=1.0,
    )


def checkpoint_path(out_dir: Path, scheduler: str) -> Path:
    return out_dir / f"checkpoint_k128_p10000000_{scheduler}.pt"


def train_or_load(
    cfg: ParticipationConfig,
    scheduler: str,
    out_dir: Path,
    train_ds: TensorDataset,
    force_retrain: bool,
) -> LatinEnsembleLM:
    torch.manual_seed(cfg.seed)
    run_device = device()
    model = build_model(cfg, scheduler).to(run_device)
    ckpt_path = checkpoint_path(out_dir, scheduler)
    if ckpt_path.exists() and not force_retrain:
        state = torch.load(ckpt_path, map_location=run_device, weights_only=True)
        model.load_state_dict(state["model"])
        return model

    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    micro_batch_size = cfg.batch_size // cfg.grad_accum_steps
    pin_memory = run_device.type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    opt_kwargs: dict[str, Any] = {"fused": True} if run_device.type == "cuda" else {}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, **opt_kwargs
    )
    use_amp = run_device.type == "cuda"
    rows: list[dict[str, float | int]] = []
    for epoch in range(cfg.epochs):
        model.train()
        micro_iter = iter(loader)
        iterator = tqdm(
            range(cfg.max_train_batches),
            desc=f"participation train {scheduler} e={epoch}",
            mininterval=10,
        )
        for step in iterator:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                try:
                    x, y = next(micro_iter)
                except StopIteration:
                    micro_iter = iter(loader)
                    x, y = next(micro_iter)
                x = x.to(run_device, non_blocking=pin_memory)
                y = y.to(run_device, non_blocking=pin_memory)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = ensemble_loss_from_hidden(
                        model, x, y, vocab_chunk_size=cfg.vocab_chunk_size
                    )
                    (loss / cfg.grad_accum_steps).backward()
                total_loss += float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss = total_loss / cfg.grad_accum_steps
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "update": epoch * cfg.max_train_batches + step,
                    "train_loss": train_loss,
                }
            )
            iterator.set_postfix(loss=f"{train_loss:.3f}")

    torch.save(
        {
            "config": asdict(cfg),
            "scheduler": scheduler,
            "model": model.state_dict(),
            "d_model": model.d_model,
            "actual_trainable_params": count_parameters(model),
        },
        ckpt_path,
    )
    curve_path = out_dir / f"training_curve_k128_p10000000_{scheduler}.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "update", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)
    return model


def subset_sizes(k: int) -> list[int]:
    return [1, 8, *range(16, k + 1, 8)]


def sample_subsets(k: int, size: int, sample_count: int, seed: int) -> list[list[int]]:
    if size == k:
        return [list(range(k))]
    if size == 1:
        # Cover a stable representative spread rather than only random singleton columns.
        count = min(sample_count, k)
        return [list(item) for item in torch.linspace(0, k - 1, count).round().long().unique().view(-1, 1).tolist()]
    generator = torch.Generator().manual_seed(seed + 1009 * size)
    subsets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(subsets) < sample_count:
        subset = tuple(sorted(torch.randperm(k, generator=generator)[:size].tolist()))
        if subset in seen:
            continue
        seen.add(subset)
        subsets.append(list(subset))
    return subsets


@torch.no_grad()
def selected_ensemble_loss(
    model: LatinEnsembleLM,
    input_ids: Tensor,
    labels: Tensor,
    active_columns: Tensor,
    vocab_chunk_size: int,
) -> Tensor:
    hidden = model.hidden_paths(input_ids, active_columns=active_columns)
    target_emb = model.token_emb(labels)
    target_logits = (hidden * target_emb[:, None, :, :]).sum(dim=-1)
    log_denom: Tensor | None = None
    weight = model.token_emb.weight
    for start in range(0, weight.shape[0], vocab_chunk_size):
        logits = F.linear(hidden, weight[start : start + vocab_chunk_size])
        chunk_lse = logits.logsumexp(dim=-1)
        log_denom = chunk_lse if log_denom is None else torch.logaddexp(log_denom, chunk_lse)
    if log_denom is None:
        raise ValueError("vocabulary must not be empty")
    member_log_probs = target_logits - log_denom
    ensemble_log_probs = torch.logsumexp(member_log_probs, dim=1) - math.log(
        active_columns.numel()
    )
    return -ensemble_log_probs.mean()


@torch.no_grad()
def evaluate_subset(
    cfg: ParticipationConfig,
    model: LatinEnsembleLM,
    val_loader: DataLoader,
    columns: list[int],
) -> float:
    run_device = next(model.parameters()).device
    pin_memory = run_device.type == "cuda"
    active_columns = torch.tensor(columns, device=run_device, dtype=torch.long)
    total = 0.0
    count = 0
    use_amp = run_device.type == "cuda"
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.max_eval_batches:
            break
        x = x.to(run_device, non_blocking=pin_memory)
        y = y.to(run_device, non_blocking=pin_memory)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = selected_ensemble_loss(
                model,
                x,
                y,
                active_columns=active_columns,
                vocab_chunk_size=cfg.vocab_chunk_size,
            )
        batch_tokens = x.numel()
        total += float(loss.detach().cpu()) * batch_tokens
        count += batch_tokens
    return total / max(1, count)


def run_participation_analysis(
    cfg: ParticipationConfig,
    out_dir: Path,
    force_retrain: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_wikitext(cfg.vocab_size, cfg.seq_len)
    run_device = device()
    pin_memory = run_device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    rows: list[dict[str, Any]] = []
    for scheduler in ("max", "min"):
        model = train_or_load(cfg, scheduler, out_dir, train_ds, force_retrain)
        model.to(run_device)
        for size in subset_sizes(cfg.k):
            subsets = sample_subsets(cfg.k, size, cfg.subset_samples, cfg.seed)
            for subset_id, columns in enumerate(
                tqdm(subsets, desc=f"eval {scheduler} columns={size}", mininterval=10)
            ):
                loss = evaluate_subset(cfg, model, val_loader, columns)
                rows.append(
                    {
                        "scheduler": scheduler,
                        "column_count": size,
                        "subset_id": subset_id,
                        "columns": " ".join(str(column) for column in columns),
                        "ensemble_loss": loss,
                    }
                )
        del model
        if run_device.type == "cuda":
            torch.cuda.empty_cache()
    detail_path = out_dir / "participation_loss_samples.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scheduler",
                "column_count",
                "subset_id",
                "columns",
                "ensemble_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows: list[dict[str, Any]], out_dir: Path, cfg: ParticipationConfig) -> None:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["scheduler"], int(row["column_count"])), []).append(
            float(row["ensemble_loss"])
        )
    summary_rows = []
    for (scheduler, size), losses in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        mean = sum(losses) / len(losses)
        low = min(losses)
        high = max(losses)
        summary_rows.append(
            {
                "scheduler": scheduler,
                "column_count": size,
                "sample_count": len(losses),
                "mean_loss": mean,
                "min_loss": low,
                "max_loss": high,
                "range_loss": high - low,
            }
        )
    summary_path = out_dir / "participation_loss_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scheduler",
                "column_count",
                "sample_count",
                "mean_loss",
                "min_loss",
                "max_loss",
                "range_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    write_static_plot(summary_rows, out_dir)
    write_interactive_plot(summary_rows, out_dir)
    write_report(summary_rows, out_dir, cfg)


def write_static_plot(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    for scheduler in ("max", "min"):
        rows = [row for row in summary_rows if row["scheduler"] == scheduler]
        x = [row["column_count"] for row in rows]
        y = [row["mean_loss"] for row in rows]
        yerr = [
            [row["mean_loss"] - row["min_loss"] for row in rows],
            [row["max_loss"] - row["mean_loss"] for row in rows],
        ]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=scheduler)
    plt.xlabel("Number of columns in ensemble sample")
    plt.ylabel("Validation ensemble loss")
    plt.title("k=128, 10M Latin model: loss vs column participation")
    plt.legend(title="Scheduler")
    plt.tight_layout()
    plt.savefig(out_dir / "participation_loss_vs_columns.png", dpi=170)
    plt.close()


def write_interactive_plot(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    traces = []
    for scheduler in ("max", "min"):
        rows = [row for row in summary_rows if row["scheduler"] == scheduler]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": scheduler,
                "x": [row["column_count"] for row in rows],
                "y": [row["mean_loss"] for row in rows],
                "error_y": {
                    "type": "data",
                    "symmetric": False,
                    "array": [row["max_loss"] - row["mean_loss"] for row in rows],
                    "arrayminus": [row["mean_loss"] - row["min_loss"] for row in rows],
                    "visible": True,
                    "thickness": 1.2,
                    "width": 4,
                },
                "customdata": [
                    [row["min_loss"], row["max_loss"], row["range_loss"], row["sample_count"]]
                    for row in rows
                ],
                "hovertemplate": (
                    "scheduler=%{fullData.name}<br>"
                    "columns=%{x}<br>"
                    "mean loss=%{y:.4f}<br>"
                    "min loss=%{customdata[0]:.4f}<br>"
                    "max loss=%{customdata[1]:.4f}<br>"
                    "range=%{customdata[2]:.4f}<br>"
                    "subsets=%{customdata[3]}<extra></extra>"
                ),
            }
        )
    layout = {
        "title": "k=128, 10M Latin model: loss vs column participation",
        "xaxis": {"title": "Number of columns in ensemble sample"},
        "yaxis": {"title": "Validation ensemble loss"},
        "hovermode": "closest",
        "template": "plotly_white",
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Participation Loss vs Columns</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    Plotly.newPlot("plot", {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});
  </script>
</body>
</html>
"""
    (out_dir / "participation_loss_vs_columns.html").write_text(html)


def write_report(
    summary_rows: list[dict[str, Any]],
    out_dir: Path,
    cfg: ParticipationConfig,
) -> None:
    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# Participation Analysis",
        "",
        "This diagnostic retrains and evaluates the `k=128`, `10M` Latin-square model with all columns active, once with the max-entropy scheduler and once with the min-entropy scheduler.",
        "",
        "For each scheduler, it evaluates sampled column subsets at sizes `1, 8, 16, 24, ..., 128`. The plotted point is the average validation ensemble loss across sampled subsets of that size. The vertical range bar spans the best to worst subset loss at the same size, so it measures how sensitive the ensemble is to which columns are chosen.",
        "",
        "## Scope",
        "",
        f"- Training exposure per scheduler: `{cfg.epochs} * {cfg.max_train_batches}` optimizer updates, effective batch `{cfg.batch_size}`, sequence length `{cfg.seq_len}`.",
        f"- Evaluation exposure per subset: `{cfg.max_eval_batches}` validation batches with batch size 1.",
        f"- Random column subsets per size: `{cfg.subset_samples}`, except the full 128-column point has one subset.",
        "- The ensemble loss is computed by averaging member probabilities post-softmax, not by averaging logits.",
        "",
        "## Files",
        "",
        "- `participation_loss_vs_columns.html`: interactive plot with min/max range bars.",
        "- `participation_loss_vs_columns.png`: static plot.",
        "- `participation_loss_summary.csv`: one row per scheduler and subset size.",
        "- `participation_loss_samples.csv`: one row per sampled column subset.",
        "- `checkpoint_k128_p10000000_{max,min}.pt`: saved model checkpoints for reruns.",
        "",
        "## Summary Table",
        "",
        "| scheduler | columns | subsets | mean_loss | min_loss | max_loss | range |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scheduler"]),
                    str(row["column_count"]),
                    str(row["sample_count"]),
                    fmt(float(row["mean_loss"])),
                    fmt(float(row["min_loss"])),
                    fmt(float(row["max_loss"])),
                    fmt(float(row["range_loss"])),
                ]
            )
            + " |"
        )
    (out_dir / "WRITEUP.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/participation_k128_10m"))
    parser.add_argument("--max-eval-batches", type=int, default=128)
    parser.add_argument("--subset-samples", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    cfg = ParticipationConfig(
        max_eval_batches=args.max_eval_batches,
        subset_samples=args.subset_samples,
        num_workers=args.num_workers,
    )
    rows = run_participation_analysis(cfg, args.out_dir, args.force_retrain)
    summarize(rows, args.out_dir, cfg)


if __name__ == "__main__":
    main()
