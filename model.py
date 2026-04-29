from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def submodel_width(total_params: int, k: int, heads: int = 2, ff_mult: int = 4) -> int:
    """Choose a small transformer width with roughly total_params across K blocks."""
    # One block has approximately 4*d^2 attention params plus 2*ff_mult*d^2 MLP params.
    coeff = 4 + 2 * ff_mult
    d_model = int(math.sqrt(total_params / max(1, k * coeff)))
    d_model = max(heads * 8, (d_model // heads) * heads)
    return d_model


class SubModel(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=causal_mask, need_weights=False)
        x = x + y
        return x + self.mlp(self.ln2(x))


class LatinEnsembleLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        schedule: list[list[int]],
        d_model: int,
        n_heads: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.schedule = schedule
        self.k = len(schedule)
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [SubModel(d_model, n_heads, dropout) for _ in range(self.k)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        paths = x[:, None, :, :].expand(batch, self.k, seq_len, self.d_model)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        outputs = []
        for col in range(self.k):
            y = paths[:, col]
            for row in range(self.k):
                y = self.blocks[self.schedule[row][col]](y, causal_mask)
            outputs.append(self.final_ln(y))

        hidden = torch.stack(outputs, dim=1)
        return F.linear(hidden, self.token_emb.weight)


def ensemble_loss(logits: Tensor, labels: Tensor) -> Tensor:
    probs = logits.softmax(dim=-1).mean(dim=1).clamp_min(1e-12)
    return F.nll_loss(probs.log().flatten(0, 1), labels.flatten())


@torch.no_grad()
def ensemble_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    probs = logits.softmax(dim=-1)
    mean_probs = probs.mean(dim=1).clamp_min(1e-12)
    ensemble_ce = F.nll_loss(mean_probs.log().flatten(0, 1), labels.flatten()).item()
    member_losses = [
        F.cross_entropy(logits[:, i].flatten(0, 1), labels.flatten()).item()
        for i in range(logits.shape[1])
    ]
    avg_member_ce = float(sum(member_losses) / len(member_losses))
    certainty = probs.max(dim=-1).values.mean(dim=(0, 2))
    mean_member = probs.mean(dim=1, keepdim=True).clamp_min(1e-12)
    kl_to_ensemble = (probs * (probs.clamp_min(1e-12).log() - mean_member.log())).sum(
        dim=-1
    )
    return {
        "ensemble_loss": ensemble_ce,
        "avg_individual_loss": avg_member_ce,
        "ensemble_gain": avg_member_ce - ensemble_ce,
        "member_certainty_mean": certainty.mean().item(),
        "member_certainty_min": certainty.min().item(),
        "member_certainty_max": certainty.max().item(),
        "member_certainty_std": certainty.std(unbiased=False).item(),
        "prediction_dissimilarity_kl": kl_to_ensemble.mean().item(),
    }
