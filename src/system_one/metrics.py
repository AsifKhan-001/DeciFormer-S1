import numpy as np
import torch


def calculate_entropy(probs: torch.Tensor, num_options=None, eps: float = 1e-12) -> torch.Tensor:
    """Entropy [B,1]; divided by log(K) (K = valid options) when num_options is given -> range [0,1]."""
    ent = -(probs * torch.log(probs + eps)).sum(-1, keepdim=True)
    if num_options is None:
        return ent
    n = torch.as_tensor(num_options, dtype=ent.dtype, device=ent.device)
    n = n.view(-1, 1) if n.dim() > 0 else n
    return ent / torch.log(n.clamp(min=2))


def calculate_top2_margin(probs: torch.Tensor) -> torch.Tensor:
    top = probs.topk(2, dim=-1).values
    return top[:, :1] - top[:, 1:2]


def confusion_matrix(target, pred, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (target, pred), 1)
    return cm


def macro_prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    present = cm.sum(1) > 0
    return float(p[present].mean()), float(r[present].mean()), float(f[present].mean())


def calculate_ece(conf, correct, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        n = int(m.sum())
        acc = float(correct[m].mean()) if n else None
        c = float(conf[m].mean()) if n else None
        if n:
            total += n / len(conf) * abs(acc - c)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "samples": n, "accuracy": acc, "confidence": c})
    return float(total), table


def calculate_brier_score(probs, target) -> float:
    onehot = np.eye(probs.shape[1])[target]
    return float(((probs - onehot) ** 2).sum(1).mean())


def calculate_nll(probs, target) -> float:
    return float(-np.log(probs[np.arange(len(target)), target] + 1e-12).mean())


def selective_table(conf, correct, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    rows = []
    for t in thresholds:
        m = conf >= t
        cov = float(m.mean())
        acc = float(correct[m].mean()) if m.any() else None
        rows.append({"threshold": t, "coverage": cov, "accuracy": acc,
                     "risk": None if acc is None else 1 - acc, "abstention_rate": 1 - cov})
    return rows


def quick_metrics(arr: dict) -> dict:
    probs, target, pred = arr["probs"], arr["target"], arr["pred"]
    correct = (pred == target).astype(float)
    cm = confusion_matrix(target, pred, probs.shape[1])
    return {
        "val_loss": float(arr.get("loss", float("nan"))),
        "val_acc": float(correct.mean()),
        "val_f1": macro_prf(cm)[2],
        "val_ece": calculate_ece(arr["conf"], correct)[0],
        "val_brier": calculate_brier_score(probs, target),
        "val_nll": calculate_nll(probs, target),
    }


def classification_report(arr: dict, labels: dict, policy, difficulty=None, bins: int = 10) -> dict:
    probs, target, pred = arr["probs"], arr["target"], arr["pred"]
    correct = (pred == target).astype(float)
    k = probs.shape[1]
    cm = confusion_matrix(target, pred, k)
    p, r, f = macro_prf(cm)
    ece, table = calculate_ece(arr["conf"], correct, bins)
    ece_soft, _ = calculate_ece(arr["maxprob"], correct, bins)
    actions = np.array(policy(arr["conf"], arr["margin"], arr["entropy"]))
    route = actions == "ROUTE"
    rep = {
        "num_samples": int(len(target)),
        "accuracy": float(correct.mean()),
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "nll": calculate_nll(probs, target),
        "brier": calculate_brier_score(probs, target),
        "ece": ece, "ece_softmax": ece_soft,
        "avg_confidence": float(arr["conf"].mean()),
        "confusion_matrix": cm.tolist(),
        "class_names": [labels.get(i, str(i)) for i in range(k)],
        "calibration_bins": table,
        "selective": selective_table(arr["conf"], correct),
        "policy": {
            "route_rate": float(route.mean()),
            "auto_routing_accuracy": float(correct[route].mean()) if route.any() else None,
            "wrong_auto_route_rate": float((route & (correct == 0)).mean()),
            "clarify_rate": float((actions == "CLARIFY").mean()),
            "abstain_rate": float((actions == "ABSTAIN").mean()),
            "escalate_rate": float((actions == "ESCALATE").mean()),
        },
    }
    if difficulty is not None:
        by = {}
        for d in sorted({x for x in difficulty if x}):
            m = np.array([x == d for x in difficulty])
            by[d] = {"n": int(m.sum()), "accuracy": float(correct[m].mean())}
        rep["by_difficulty"] = by
    return rep


def save_plots(arr: dict, rep: dict, out_dir) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm = np.array(rep["confusion_matrix"])
    names = rep["class_names"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    rows = [r for r in rep["calibration_bins"] if r["samples"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot([r["confidence"] for r in rows], [r["accuracy"] for r in rows], "o-", label=f"ECE={rep['ece']:.3f}")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(arr["conf"], bins=20, range=(0, 1))
    ax.set_xlabel("confidence")
    ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(out / "confidence_histogram.png", dpi=150)
    plt.close(fig)

    sel = [r for r in rep["selective"] if r["accuracy"] is not None]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["coverage"] for r in sel], [r["accuracy"] for r in sel], "o-")
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_accuracy.png", dpi=150)
    plt.close(fig)
