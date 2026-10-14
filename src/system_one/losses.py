import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)


class BrierLoss(nn.Module):
    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(target, probs.size(-1)).to(probs.dtype)
        return ((probs - onehot) ** 2).sum(-1).mean()


class ConfidenceLoss(nn.Module):
    def forward(self, confidence: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        correct = (prediction == target).float().unsqueeze(1)
        return F.binary_cross_entropy(confidence.clamp(1e-6, 1 - 1e-6), correct)


class TotalLoss(nn.Module):
    """CE(scaled logits) + alpha * Brier + beta * confidence BCE."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.ce, self.brier, self.conf = ClassificationLoss(), BrierLoss(), ConfidenceLoss()

    def forward(self, out, target: torch.Tensor):
        ce = self.ce(out.scaled_logits, target)
        brier = self.brier(out.probabilities, target)
        conf = self.conf(out.confidence, out.prediction, target) if out.confidence is not None else ce.new_zeros(())
        total = ce + self.alpha * brier + self.beta * conf
        return total, {"ce": ce.item(), "brier": brier.item(), "conf": conf.item()}
