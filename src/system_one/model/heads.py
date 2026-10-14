import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionPooling(nn.Module):
    """hidden [B,N,d] -> [B,d]"""

    def __init__(self, mode: str, d: int):
        super().__init__()
        self.mode = mode
        self.score = nn.Linear(d, 1) if mode == "attention" else None

    def forward(self, h, attention_mask, position=None):
        if self.mode in ("last", "decision_token"):
            pos = position if position is not None else attention_mask.sum(1) - 1
            return h[torch.arange(h.size(0), device=h.device), pos]
        if self.mode == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        s = self.score(h).squeeze(-1).masked_fill(attention_mask == 0, float("-inf"))
        return (s.softmax(-1).unsqueeze(-1) * h).sum(1)


class DecisionHead(nn.Module):
    """[B,d] -> logits [B,K]"""

    def __init__(self, d: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d // 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptionScorer(nn.Module):
    """Shared scorer: [B,K,d] -> [B,K]"""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class TemperatureScaler(nn.Module):
    """T = softplus(raw) + min_temperature (always positive)."""

    def __init__(self, init: float = 1.0, min_temperature: float = 0.05, learnable: bool = True):
        super().__init__()
        self.min_t = min_temperature
        raw = torch.tensor(_inv_softplus(max(init - min_temperature, 1e-4)))
        self.raw = nn.Parameter(raw, requires_grad=learnable)

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw) + self.min_t

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def set_temperature(self, t: float) -> None:
        self.raw.fill_(_inv_softplus(max(t - self.min_t, 1e-4)))


class CalibrationHead(nn.Module):
    """[B,d] + entropy [B,1] -> confidence [B,1]"""

    def __init__(self, d: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, rep: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([rep, entropy], dim=-1)))
