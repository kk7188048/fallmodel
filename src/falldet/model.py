"""M3 - BiLSTM with attention pooling for fall classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learns a scalar importance score per timestep and returns the
    weighted sum of the sequence, so the classifier can focus on the
    frames that actually look like a fall instead of averaging them away
    with the many upright/idle frames that surround it in a window.
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: (batch, time, input_dim)
        scores = self.score(sequence).squeeze(-1)  # (batch, time)
        weights = torch.softmax(scores, dim=-1)  # (batch, time)
        pooled = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)  # (batch, input_dim)
        return pooled, weights


class FallDetectionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        attention_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # dropout between LSTM layers only applies when num_layers > 1;
            # PyTorch warns (harmlessly) if set with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = AttentionPooling(lstm_out_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (batch, time, input_dim) -> logits: (batch, num_classes)
        sequence_out, _ = self.lstm(x)
        pooled, weights = self.attention(sequence_out)
        logits = self.classifier(self.dropout(pooled))
        if return_attention:
            return logits, weights
        return logits

    @classmethod
    def from_config(cls, config: dict, input_dim: int) -> "FallDetectionModel":
        model_cfg = config["model"]
        return cls(
            input_dim=input_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg["bidirectional"],
            attention_dim=model_cfg["attention_dim"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"],
        )
