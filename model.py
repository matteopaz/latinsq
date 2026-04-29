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


class ParallelSubModels(nn.Module):
    def __init__(self, k: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.k = k
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.ln1_weight = nn.Parameter(torch.ones(k, d_model))
        self.ln1_bias = nn.Parameter(torch.zeros(k, d_model))
        self.qkv_weight = nn.Parameter(torch.empty(k, 3 * d_model, d_model))
        self.qkv_bias = nn.Parameter(torch.zeros(k, 3 * d_model))
        self.out_weight = nn.Parameter(torch.empty(k, d_model, d_model))
        self.out_bias = nn.Parameter(torch.zeros(k, d_model))

        self.ln2_weight = nn.Parameter(torch.ones(k, d_model))
        self.ln2_bias = nn.Parameter(torch.zeros(k, d_model))
        self.fc1_weight = nn.Parameter(torch.empty(k, 4 * d_model, d_model))
        self.fc1_bias = nn.Parameter(torch.zeros(k, 4 * d_model))
        self.fc2_weight = nn.Parameter(torch.empty(k, d_model, 4 * d_model))
        self.fc2_bias = nn.Parameter(torch.zeros(k, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (
            self.qkv_weight,
            self.out_weight,
            self.fc1_weight,
            self.fc2_weight,
        ):
            for sub_weight in weight:
                nn.init.xavier_uniform_(sub_weight)

    def _layer_norm(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        y = F.layer_norm(x, (self.d_model,))
        return y * weight[None, :, None, :] + bias[None, :, None, :]

    def _linear(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        x = x.transpose(0, 1)
        y = torch.bmm(x.flatten(1, 2), weight.transpose(1, 2))
        y = y.view(self.k, -1, x.shape[2], weight.shape[1]).transpose(0, 1)
        return y + bias[None, :, None, :]

    def forward(self, x: Tensor) -> Tensor:
        batch, k, seq_len, _ = x.shape
        if k != self.k:
            raise ValueError(f"expected {self.k} simultaneous paths, got {k}")

        y = self._layer_norm(x, self.ln1_weight, self.ln1_bias)
        qkv = self._linear(y, self.qkv_weight, self.qkv_bias)
        q, key, value = qkv.chunk(3, dim=-1)
        q = q.reshape(batch * k, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch * k, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch * k, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(batch, k, seq_len, self.d_model)
        x = x + self._linear(y, self.out_weight, self.out_bias)

        y = self._layer_norm(x, self.ln2_weight, self.ln2_bias)
        y = F.gelu(self._linear(y, self.fc1_weight, self.fc1_bias))
        y = self._linear(y, self.fc2_weight, self.fc2_bias)
        y = F.dropout(y, p=self.dropout, training=self.training)
        return x + y


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
        self.k = len(schedule)
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        schedule_tensor = torch.tensor(schedule, dtype=torch.long)
        self.register_buffer("schedule", schedule_tensor)
        self.register_buffer("inverse_schedule", torch.argsort(schedule_tensor, dim=1))
        self.blocks = ParallelSubModels(self.k, d_model, n_heads, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        paths = x[:, None, :, :].expand(batch, self.k, seq_len, self.d_model).clone()
        for row in self.inverse_schedule:
            by_model = paths.index_select(1, row)
            by_model = self.blocks(by_model)
            paths = torch.empty_like(paths).index_copy(1, row, by_model)

        hidden = self.final_ln(paths)
        return F.linear(hidden, self.token_emb.weight)


def ensemble_loss(logits: Tensor, labels: Tensor) -> Tensor:
    logits = logits.float()
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
