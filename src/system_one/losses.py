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
    """CE(scaled logits) + alpha * Brier + beta * confidence BCE + order_weight * order consistency."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, order_weight: float = 0.0):
        super().__init__()
        self.alpha, self.beta, self.order_weight = alpha, beta, order_weight
        self.ce, self.brier, self.conf = ClassificationLoss(), BrierLoss(), ConfidenceLoss()

    @staticmethod
    def order_consistency(probs: torch.Tensor, perm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """probs/perm [2B,K]: first B rows original order, last B rows shuffled; perm[k] = original index of option k."""
        b = probs.size(0) // 2
        aligned = torch.zeros_like(probs).scatter(1, perm, probs)
        p, q = aligned[:b], aligned[b:]
        kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)
        kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(-1)
        return 0.5 * (kl_pq + kl_qp).mean()

    def forward(self, out, target: torch.Tensor, perm: torch.Tensor = None):
        ce = self.ce(out.scaled_logits, target)
        brier = self.brier(out.probabilities, target)
        conf = self.conf(out.confidence, out.prediction, target) if out.confidence is not None else ce.new_zeros(())
        order = ce.new_zeros(())
        if perm is not None:
            order = self.order_consistency(out.probabilities, perm)
        total = ce + self.alpha * brier + self.beta * conf + self.order_weight * order
        return total, {"ce": ce.item(), "brier": brier.item(), "conf": conf.item(), "order": order.item()}
