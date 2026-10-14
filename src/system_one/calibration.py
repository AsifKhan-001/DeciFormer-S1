import torch
import torch.nn.functional as F

from .trainer import collect


def fit_temperature(raw_logits, target, min_t: float = 0.05) -> float:
    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    tgt = torch.as_tensor(target, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), tgt)
        loss.backward()
        return loss

    opt.step(closure)
    return max(float(log_t.exp().detach()), min_t + 1e-3)


def refit_calibration_head(model, loader, device, steps: int = 300, lr: float = 1e-3) -> None:
    arr = collect(model, loader, device, keep_rep=True)
    rep = torch.as_tensor(arr["rep"]).to(device)
    ent = torch.as_tensor(arr["entropy"]).unsqueeze(1).to(device)
    correct = torch.as_tensor((arr["pred"] == arr["target"]).astype("float32")).unsqueeze(1).to(device)
    head = model.calibration_head
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        loss = F.binary_cross_entropy(head(rep, ent).clamp(1e-6, 1 - 1e-6), correct)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
