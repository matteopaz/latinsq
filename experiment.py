from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LatinEnsembleLM, ensemble_loss, ensemble_metrics, submodel_width
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


def schedule(name: str, k: int) -> list[list[int]]:
    if name == "min":
        return min_entropy_latin_square(k)
    if name == "max":
        return max_entropy_latin_square(k)
    raise ValueError(f"unknown scheduler: {name}")


def load_wikitext(vocab_size: int, seq_len: int) -> tuple[TensorDataset, TensorDataset]:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

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

    return encode("train"), encode("validation")


def synthetic_data(vocab_size: int, seq_len: int, n: int = 128) -> tuple[TensorDataset, TensorDataset]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab_size, (n, seq_len), generator=gen)
    y = torch.roll(x, shifts=-1, dims=1)
    return TensorDataset(x, y), TensorDataset(x[: n // 4], y[: n // 4])


def run_one(cfg: Config, out_dir: Path, smoke: bool = False) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = synthetic_data(
        cfg.vocab_size, cfg.seq_len
    ) if smoke else load_wikitext(cfg.vocab_size, cfg.seq_len)
    width = submodel_width(cfg.total_params, cfg.k)
    model = LatinEnsembleLM(
        vocab_size=cfg.vocab_size,
        schedule=schedule(cfg.scheduler, cfg.k),
        d_model=width,
        max_seq_len=cfg.seq_len,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    curve_path = out_dir / f"curve_k{cfg.k}_p{cfg.total_params}_{cfg.scheduler}.csv"

    rows = []
    for epoch in range(cfg.epochs):
        model.train()
        iterator = tqdm(train_loader, desc=f"k={cfg.k} p={cfg.total_params} {cfg.scheduler} e={epoch}")
        for step, (x, y) in enumerate(iterator):
            if cfg.max_train_batches is not None and step >= cfg.max_train_batches:
                break
            opt.zero_grad(set_to_none=True)
            loss = ensemble_loss(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rows.append({"epoch": epoch, "step": step, "train_loss": loss.item()})
            iterator.set_postfix(loss=f"{loss.item():.3f}")

    metrics = evaluate(model, val_loader, device, cfg.max_eval_batches)
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)
    return {**asdict(cfg), "d_model": width, **metrics}


@torch.no_grad()
def evaluate(model: LatinEnsembleLM, loader: DataLoader, device: torch.device, max_batches: int | None) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        metrics = ensemble_metrics(model(x.to(device)), y.to(device))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    return {key: value / max(1, n) for key, value in totals.items()}


def write_summary(results: list[dict[str, float | int | str]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    for params in sorted(df["total_params"].unique()):
        subset = df[df["total_params"] == params]
        for scheduler in sorted(subset["scheduler"].unique()):
            trace = subset[subset["scheduler"] == scheduler].sort_values("k")
            plt.plot(trace["k"], trace["ensemble_loss"], marker="o", label=f"{params} {scheduler}")
    plt.xlabel("K")
    plt.ylabel("Final held-out validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_loss_vs_k.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true", help="use synthetic data and one tiny run")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        configs = [
            Config(
                k=2,
                total_params=20_000,
                scheduler="min",
                epochs=1,
                max_train_batches=1,
                max_eval_batches=1,
            )
        ]
    else:
        configs = [
            Config(k=k, total_params=params, scheduler=scheduler)
            for k in (16, 64, 128)
            for params in (1_000_000, 10_000_000)
            for scheduler in ("min", "max")
        ]
    write_summary([run_one(cfg, args.out_dir, args.smoke) for cfg in configs], args.out_dir)


if __name__ == "__main__":
    main()
