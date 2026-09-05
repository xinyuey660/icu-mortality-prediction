"""PyTorch models for irregular ICU time series.

The architectures are intentionally compact for the 4,000-patient public Set A.
Larger networks are more likely to memorize this benchmark than to generalize.
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without training extra
    raise ImportError(
        "PyTorch is required for neural models. Install with: pip install '.[train]'"
    ) from exc


class StaticEncoder(nn.Module):
    def __init__(self, static_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim if static_dim else 0
        self.network = (
            nn.Sequential(
                nn.Linear(static_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            if static_dim
            else None
        )

    def forward(self, static: torch.Tensor | None) -> torch.Tensor | None:
        return self.network(static) if self.network is not None and static is not None else None


class GRUDCell(nn.Module):
    """GRU-D with feature-wise input decay and elapsed-time hidden decay."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # The paper's input decay is feature-wise (diagonal), not a dense F x F map.
        self.gamma_x_weight = nn.Parameter(torch.zeros(input_dim))
        self.gamma_x_bias = nn.Parameter(torch.zeros(input_dim))
        self.gamma_h = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mean: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, _ = values.shape
        hidden = values.new_zeros((batch_size, self.hidden_dim))
        last_observed = feature_mean.unsqueeze(0).expand(batch_size, -1).clone()
        mean = feature_mean.unsqueeze(0)
        for time_index in range(time_steps):
            x_t = values[:, time_index]
            m_t = mask[:, time_index]
            d_t = delta[:, time_index]
            gamma_x = torch.exp(
                -torch.relu(d_t * self.gamma_x_weight + self.gamma_x_bias)
            )
            gamma_h = torch.exp(-torch.relu(self.gamma_h(d_t)))
            x_hat = m_t * x_t + (1.0 - m_t) * (
                gamma_x * last_observed + (1.0 - gamma_x) * mean
            )
            hidden = gamma_h * hidden
            hidden = self.gru_cell(torch.cat([x_hat, m_t], dim=-1), hidden)
            last_observed = torch.where(m_t.bool(), x_t, last_observed)
        return hidden


class GRUDClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.cell = GRUDCell(input_dim, hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.static_encoder = StaticEncoder(static_dim, 24, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.static_encoder.output_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        representation = self.cell(values, mask, delta, self.feature_mean)
        static_representation = self.static_encoder(static)
        if static_representation is not None:
            representation = torch.cat([representation, static_representation], dim=-1)
        return self.head(representation).squeeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class RecurrentAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        recurrent_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(recurrent_dim, hidden_dim)
        self.static_encoder = StaticEncoder(static_dim, 24, dropout)
        self.head = nn.Sequential(
            nn.Linear(recurrent_dim + self.static_encoder.output_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        static: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        sequence_input = torch.cat([values, mask, delta], dim=-1)
        sequence, _ = self.gru(sequence_input)
        context, attention = self.attention(sequence)
        static_representation = self.static_encoder(static)
        if static_representation is not None:
            context = torch.cat([context, static_representation], dim=-1)
        logits = self.head(context).squeeze(-1)
        return (logits, attention) if return_attention else logits


class GRUAttentionClassifier(RecurrentAttentionClassifier):
    def __init__(self, input_dim: int, static_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__(input_dim * 3, static_dim, hidden_dim, dropout, bidirectional=False)


class BiGRUAttentionClassifier(RecurrentAttentionClassifier):
    def __init__(self, input_dim: int, static_dim: int, hidden_dim: int = 48, dropout: float = 0.30):
        super().__init__(input_dim * 3, static_dim, hidden_dim, dropout, bidirectional=True)


class ContinuousValueEmbedding(nn.Module):
    """One-to-many continuous embedding used for values and timestamps in STraTS."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        hidden = max(4, int(math.sqrt(embedding_dim)))
        self.network = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, embedding_dim)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.unsqueeze(-1))


class TripletTransformerClassifier(nn.Module):
    """STraTS-inspired observation-triplet embedding classifier.

    Each non-missing observation is encoded as feature + value + time. This is
    an independently implemented downstream classifier; it does not claim the
    paper's optional self-supervised forecasting pretraining.
    """

    def __init__(
        self,
        n_features: int,
        static_dim: int,
        embedding_dim: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads.")
        self.no_observation_id = n_features
        self.padding_id = n_features + 1
        self.feature_embedding = nn.Embedding(
            n_features + 2, embedding_dim, padding_idx=self.padding_id
        )
        self.value_embedding = ContinuousValueEmbedding(embedding_dim)
        self.time_embedding = ContinuousValueEmbedding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.fusion_query = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.fusion_query, std=0.02)
        self.static_encoder = StaticEncoder(static_dim, 24, dropout)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + self.static_encoder.output_dim, embedding_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        times: torch.Tensor,
        padding_mask: torch.Tensor,
        static: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        tokens = (
            self.feature_embedding(feature_ids)
            + self.value_embedding(values)
            + self.time_embedding(times)
        )
        contextual = self.encoder(tokens, src_key_padding_mask=padding_mask)
        scores = torch.einsum("btd,d->bt", contextual, self.fusion_query)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        representation = torch.sum(contextual * attention.unsqueeze(-1), dim=1)
        static_representation = self.static_encoder(static)
        if static_representation is not None:
            representation = torch.cat([representation, static_representation], dim=-1)
        logits = self.head(representation).squeeze(-1)
        return (logits, attention) if return_attention else logits
