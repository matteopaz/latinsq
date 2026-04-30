from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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
        path_count = x.shape[1]
        x = x.transpose(0, 1)
        y = torch.bmm(x.flatten(1, 2), weight.transpose(1, 2))
        y = y.view(path_count, -1, x.shape[2], weight.shape[1]).transpose(0, 1)
        return y + bias[None, :, None, :]

    def forward(self, x: Tensor, model_order: Tensor | None = None) -> Tensor:
        batch, k, seq_len, _ = x.shape
        if model_order is None and k != self.k:
            raise ValueError(f"expected {self.k} simultaneous paths, got {k}")
        if model_order is not None and k != model_order.numel():
            raise ValueError(
                f"expected one model index per active path, got {model_order.numel()} for {k} paths"
            )
        ln1_weight = self.ln1_weight
        ln1_bias = self.ln1_bias
        qkv_weight = self.qkv_weight
        qkv_bias = self.qkv_bias
        out_weight = self.out_weight
        out_bias = self.out_bias
        ln2_weight = self.ln2_weight
        ln2_bias = self.ln2_bias
        fc1_weight = self.fc1_weight
        fc1_bias = self.fc1_bias
        fc2_weight = self.fc2_weight
        fc2_bias = self.fc2_bias
        if model_order is not None:
            ln1_weight = ln1_weight.index_select(0, model_order)
            ln1_bias = ln1_bias.index_select(0, model_order)
            qkv_weight = qkv_weight.index_select(0, model_order)
            qkv_bias = qkv_bias.index_select(0, model_order)
            out_weight = out_weight.index_select(0, model_order)
            out_bias = out_bias.index_select(0, model_order)
            ln2_weight = ln2_weight.index_select(0, model_order)
            ln2_bias = ln2_bias.index_select(0, model_order)
            fc1_weight = fc1_weight.index_select(0, model_order)
            fc1_bias = fc1_bias.index_select(0, model_order)
            fc2_weight = fc2_weight.index_select(0, model_order)
            fc2_bias = fc2_bias.index_select(0, model_order)

        y = self._layer_norm(x, ln1_weight, ln1_bias)
        qkv = self._linear(y, qkv_weight, qkv_bias)
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
        x = x + self._linear(y, out_weight, out_bias)

        y = self._layer_norm(x, ln2_weight, ln2_bias)
        y = F.gelu(self._linear(y, fc1_weight, fc1_bias))
        y = self._linear(y, fc2_weight, fc2_bias)
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
        active_column_fraction: float = 0.25,
    ):
        super().__init__()
        if not 0.0 < active_column_fraction <= 1.0:
            raise ValueError("active_column_fraction must be in (0, 1]")
        self.k = len(schedule)
        self.d_model = d_model
        self.active_column_fraction = active_column_fraction
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        schedule_tensor = torch.tensor(schedule, dtype=torch.long)
        self.register_buffer("schedule", schedule_tensor)
        self.register_buffer("inverse_schedule", torch.argsort(schedule_tensor, dim=1))
        self.blocks = ParallelSubModels(self.k, d_model, n_heads, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def active_column_count(self) -> int:
        return max(1, math.ceil(self.k * self.active_column_fraction))

    def _active_columns(self, device: torch.device) -> Tensor:
        active_count = self.active_column_count()
        if active_count >= self.k:
            return torch.arange(self.k, device=device)
        return torch.randperm(self.k, device=device)[:active_count].sort().values

    def hidden_paths(
        self,
        input_ids: Tensor,
        active_columns: Tensor | None = None,
        return_active_columns: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch, seq_len = input_ids.shape
        if active_columns is None:
            active_columns = self._active_columns(input_ids.device)
        active_columns = active_columns.to(device=input_ids.device, dtype=torch.long)
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        paths = x[:, None, :, :].expand(
            batch, active_columns.numel(), seq_len, self.d_model
        ).clone()
        for row in self.schedule:
            paths = self.blocks(paths, model_order=row.index_select(0, active_columns))

        hidden = self.final_ln(paths)
        if return_active_columns:
            return hidden, active_columns
        return hidden

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.linear(self.hidden_paths(input_ids), self.token_emb.weight)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, key, value = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        y = self.attn(self.ln1(x))
        x = x + F.dropout(y, p=self.dropout, training=self.training)
        y = self.ln2(x)
        y = self.fc2(F.gelu(self.fc1(y)))
        return x + F.dropout(y, p=self.dropout, training=self.training)


class TraditionalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        logit_scale: float | None = None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.logit_scale = 1.0 / math.sqrt(d_model) if logit_scale is None else logit_scale
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith("weight") and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(param)
            elif name.endswith("weight"):
                nn.init.ones_(param)

        # GPT-2 residual projection scaling keeps activations stable as depth grows.
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.fc2.weight, mean=0.0, std=residual_std)

    def hidden(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        del batch
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x)

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.linear(self.hidden(input_ids), self.token_emb.weight) * self.logit_scale


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def traditional_width_for_budget(
    target_params: int,
    vocab_size: int,
    max_seq_len: int,
    n_layers: int = 2,
    n_heads: int = 2,
) -> int:
    best_width = n_heads
    best_count = 0
    for d_model in range(n_heads, 2048 + n_heads, n_heads):
        model = TraditionalTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )
        count = count_parameters(model)
        if count > target_params:
            break
        best_width = d_model
        best_count = count
        del model
    if best_count == 0:
        raise ValueError(f"target_params={target_params} is too small")
    return best_width


def traditional_parameter_count(
    vocab_size: int,
    max_seq_len: int,
    d_model: int,
    n_layers: int,
) -> int:
    embedding_params = vocab_size * d_model + max_seq_len * d_model
    block_params = n_layers * (12 * d_model * d_model + 13 * d_model)
    final_ln_params = 2 * d_model
    return embedding_params + block_params + final_ln_params


def traditional_arch_for_budget(
    target_params: int,
    vocab_size: int,
    max_seq_len: int,
) -> tuple[int, int, int]:
    best: tuple[float, int, int, int] | None = None
    for n_layers in range(2, 13):
        for n_heads in (1, 2, 4, 8):
            for d_model in range(n_heads * 16, 1025, n_heads):
                head_dim = d_model // n_heads
                if head_dim < 16 or head_dim > 128:
                    continue
                count = traditional_parameter_count(
                    vocab_size,
                    max_seq_len,
                    d_model,
                    n_layers,
                )
                if count > target_params:
                    break
                utilization = count / target_params
                depth_score = min(n_layers, 8) / 8
                head_dim_score = 1.0 - min(abs(head_dim - 64) / 64, 1.0)
                width_score = d_model / 1024
                score = utilization + 0.05 * depth_score + 0.02 * head_dim_score + 0.01 * width_score
                if best is None or score > best[0]:
                    best = (score, d_model, n_layers, n_heads)
    if best is None:
        raise ValueError(f"target_params={target_params} is too small")
    _, d_model, n_layers, n_heads = best
    return d_model, n_layers, n_heads


def lm_loss_from_hidden(
    model: TraditionalTransformerLM,
    input_ids: Tensor,
    labels: Tensor,
    vocab_chunk_size: int = 2048,
) -> Tensor:
    hidden = model.hidden(input_ids)
    target_emb = model.token_emb(labels)
    target_logits = (hidden * target_emb).sum(dim=-1) * model.logit_scale

    log_denom: Tensor | None = None
    weight = model.token_emb.weight
    for start in range(0, weight.shape[0], vocab_chunk_size):
        weight_chunk = weight[start : start + vocab_chunk_size]

        def chunk_logsumexp(hidden_arg: Tensor, weight_arg: Tensor) -> Tensor:
            return (F.linear(hidden_arg, weight_arg) * model.logit_scale).logsumexp(dim=-1)

        chunk_lse = checkpoint(
            chunk_logsumexp,
            hidden,
            weight_chunk,
            use_reentrant=False,
        )
        log_denom = chunk_lse if log_denom is None else torch.logaddexp(log_denom, chunk_lse)

    if log_denom is None:
        raise ValueError("vocabulary must not be empty")
    return -(target_logits - log_denom).mean()


@torch.no_grad()
def lm_metrics_from_hidden(
    model: TraditionalTransformerLM,
    input_ids: Tensor,
    labels: Tensor,
    vocab_chunk_size: int = 2048,
) -> dict[str, float]:
    hidden = model.hidden(input_ids)
    target_emb = model.token_emb(labels)
    target_logits = (hidden * target_emb).sum(dim=-1) * model.logit_scale

    log_denom: Tensor | None = None
    max_logits: Tensor | None = None
    weight = model.token_emb.weight
    for start in range(0, weight.shape[0], vocab_chunk_size):
        logits = F.linear(hidden, weight[start : start + vocab_chunk_size]) * model.logit_scale
        chunk_lse = logits.logsumexp(dim=-1)
        chunk_max = logits.max(dim=-1).values
        log_denom = chunk_lse if log_denom is None else torch.logaddexp(log_denom, chunk_lse)
        max_logits = chunk_max if max_logits is None else torch.maximum(max_logits, chunk_max)

    if log_denom is None or max_logits is None:
        raise ValueError("vocabulary must not be empty")

    loss = -(target_logits - log_denom).mean().item()
    certainty = (max_logits - log_denom).exp().mean().item()
    return {
        "validation_loss": loss,
        "certainty_mean": certainty,
    }


def ensemble_loss_from_hidden(
    model: LatinEnsembleLM,
    input_ids: Tensor,
    labels: Tensor,
    vocab_chunk_size: int = 2048,
) -> Tensor:
    hidden = model.hidden_paths(input_ids)
    target_emb = model.token_emb(labels)
    target_logits = (hidden * target_emb[:, None, :, :]).sum(dim=-1)

    log_denom: Tensor | None = None
    weight = model.token_emb.weight
    for start in range(0, weight.shape[0], vocab_chunk_size):
        weight_chunk = weight[start : start + vocab_chunk_size]

        def chunk_logsumexp(hidden_arg: Tensor, weight_arg: Tensor) -> Tensor:
            return F.linear(hidden_arg, weight_arg).logsumexp(dim=-1)

        chunk_lse = checkpoint(
            chunk_logsumexp,
            hidden,
            weight_chunk,
            use_reentrant=False,
        )
        log_denom = chunk_lse if log_denom is None else torch.logaddexp(log_denom, chunk_lse)

    if log_denom is None:
        raise ValueError("vocabulary must not be empty")
    member_log_probs = target_logits - log_denom
    ensemble_log_probs = torch.logsumexp(member_log_probs, dim=1) - math.log(
        hidden.shape[1]
    )
    return -ensemble_log_probs.mean()


@torch.no_grad()
def ensemble_metrics_from_hidden(
    model: LatinEnsembleLM,
    input_ids: Tensor,
    labels: Tensor,
    vocab_chunk_size: int = 2048,
) -> dict[str, float | list[float] | list[int]]:
    hidden, active_columns = model.hidden_paths(input_ids, return_active_columns=True)
    target_emb = model.token_emb(labels)
    target_logits = (hidden * target_emb[:, None, :, :]).sum(dim=-1)

    log_denom: Tensor | None = None
    max_logits: Tensor | None = None
    weight = model.token_emb.weight
    for start in range(0, weight.shape[0], vocab_chunk_size):
        logits = F.linear(hidden, weight[start : start + vocab_chunk_size])
        chunk_lse = logits.logsumexp(dim=-1)
        chunk_max = logits.max(dim=-1).values
        log_denom = chunk_lse if log_denom is None else torch.logaddexp(log_denom, chunk_lse)
        max_logits = chunk_max if max_logits is None else torch.maximum(max_logits, chunk_max)

    if log_denom is None or max_logits is None:
        raise ValueError("vocabulary must not be empty")

    member_log_probs = target_logits - log_denom
    ensemble_log_probs = torch.logsumexp(member_log_probs, dim=1) - math.log(
        hidden.shape[1]
    )
    ensemble_ce = -ensemble_log_probs.mean().item()
    avg_member_ce = -member_log_probs.mean().item()

    certainty = (max_logits - log_denom).exp().mean(dim=(0, 2))
    kl_total = hidden.new_tensor(0.0)
    for start in range(0, weight.shape[0], vocab_chunk_size):
        logits = F.linear(hidden, weight[start : start + vocab_chunk_size])
        log_probs = logits - log_denom[..., None]
        probs = log_probs.exp()
        mean_probs = probs.mean(dim=1).clamp_min(1e-12)
        kl_total += (
            probs
            * (log_probs - mean_probs[:, None, :, :].log())
        ).sum()

    prediction_dissimilarity_kl = (
        kl_total / (hidden.shape[0] * hidden.shape[1] * hidden.shape[2])
    ).item()
    return {
        "ensemble_loss": ensemble_ce,
        "avg_individual_loss": avg_member_ce,
        "ensemble_gain": avg_member_ce - ensemble_ce,
        "member_certainty_mean": certainty.mean().item(),
        "member_certainty_min": certainty.min().item(),
        "member_certainty_max": certainty.max().item(),
        "member_certainty_std": certainty.std(unbiased=False).item(),
        "member_certainty_by_index": certainty.detach().cpu().tolist(),
        "active_column_indices": active_columns.detach().cpu().tolist(),
        "prediction_dissimilarity_kl": prediction_dissimilarity_kl,
    }


def ensemble_loss(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels[:, None, :, None].expand(-1, logits.shape[1], -1, -1)
    target_logits = logits.gather(dim=-1, index=labels).squeeze(-1)
    member_log_probs = target_logits - logits.logsumexp(dim=-1)
    ensemble_log_probs = torch.logsumexp(member_log_probs, dim=1) - math.log(
        logits.shape[1]
    )
    return -ensemble_log_probs.mean()


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
        "member_certainty_by_index": certainty.detach().cpu().tolist(),
        "prediction_dissimilarity_kl": kl_to_ensemble.mean().item(),
    }
