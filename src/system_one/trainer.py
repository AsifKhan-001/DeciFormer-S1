import contextlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import DecisionDataset, make_collate, read_jsonl, record_texts
from .losses import TotalLoss
from .metrics import quick_metrics
from .model.system_one import SystemOneModel
from .tokenizer import build_tokenizer
from .utils import count_parameters, env_info, get_device, load_labels, set_seed

MODEL_KEYS = ("input_ids", "attention_mask", "option_indices", "option_mask", "decision_position")
LOWER_BETTER = {"val_loss", "val_ece", "val_brier", "val_nll"}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def model_inputs(batch: dict) -> dict:
    return {k: batch[k] for k in MODEL_KEYS if k in batch}


def amp_ctx(device, name: str):
    if name == "none" or device.type == "mps":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, name))


@torch.no_grad()
def collect(model, loader, device, criterion=None, keep_rep: bool = False) -> dict:
    model.eval()
    k = model.num_outputs
    keys = ["probs", "raw_logits", "conf", "maxprob", "entropy", "margin", "target", "pred"]
    acc = {key: [] for key in keys + (["rep"] if keep_rep else [])}
    loss_sum, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(**model_inputs(batch))
        tgt = batch["target"]
        if criterion is not None:
            loss_sum += criterion(out, tgt)[0].item() * tgt.numel()
        n += tgt.numel()
        pad = k - out.probabilities.size(1)
        maxprob = out.probabilities.max(-1).values
        acc["probs"].append(F.pad(out.probabilities, (0, pad)).cpu())
        acc["raw_logits"].append(F.pad(out.logits, (0, pad), value=-1e9).cpu())
        acc["conf"].append((out.confidence.squeeze(1) if out.confidence is not None else maxprob).cpu())
        acc["maxprob"].append(maxprob.cpu())
        acc["entropy"].append(out.entropy.squeeze(1).cpu())
        acc["margin"].append(out.top2_margin.squeeze(1).cpu())
        acc["target"].append(tgt.cpu())
        acc["pred"].append(out.prediction.cpu())
        if keep_rep:
            acc["rep"].append(out.representation.cpu())
    arr = {key: torch.cat(v).numpy() for key, v in acc.items()}
    arr["loss"] = loss_sum / max(n, 1)
    return arr


def build_optimizer(model, tc):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            lr = tc.lora_lr if "lora_" in name else tc.backbone_lr
        else:
            lr = tc.lr
        wd = tc.weight_decay if p.ndim >= 2 else 0.0
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW(
        [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()],
        betas=tc.betas, eps=tc.adam_eps)


def build_scheduler(opt, total_steps: int, warmup_ratio: float):
    warm = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def save_checkpoint(path, model, cfg: Config, labels: dict, state=None, optimizer=None, scheduler=None, stage=0):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": state or {}, "stage": stage,
        "config": cfg.to_dict(), "labels": labels,
        "temperature": float(model.temperature.temperature.detach()),
        "tokenizer": {"kind": cfg.data.tokenizer, "dir": "tokenizer"},
    }, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def train(model, cfg: Config, train_loader, val_loader, device, out_dir, labels, resume_path=None) -> dict:
    tc, out_dir = cfg.training, Path(out_dir)
    criterion = TotalLoss(tc.alpha_brier, tc.beta_conf)
    stages = []
    if tc.head_warmup_epochs > 0:
        stages.append((tc.head_warmup_epochs, 0.0))
    if tc.epochs > tc.head_warmup_epochs:
        stages.append((tc.epochs - tc.head_warmup_epochs, tc.unfreeze_fraction))
    bounds = np.cumsum([e for e, _ in stages]).tolist()
    stage_of = lambda ep: next(i for i, b in enumerate(bounds) if ep < b)

    state = {"epoch": 0, "step": 0, "best": None, "best_epoch": None, "bad": 0, "history": []}
    resume = None
    if resume_path:
        resume = load_checkpoint(resume_path)
        model.load_state_dict(resume["model"])
        state.update(resume["state"])
        print(f"resumed from {resume_path} at epoch {state['epoch']}, step {state['step']}")

    steps_per_epoch = math.ceil(len(train_loader) / tc.grad_accum_steps)
    lower = tc.monitor in LOWER_BETTER
    current, opt, sched = -1, None, None

    for epoch in range(state["epoch"], tc.epochs):
        s = stage_of(epoch)
        if s != current:
            current = s
            n_ep, frac = stages[s]
            model.set_backbone_trainable(frac)
            opt = build_optimizer(model, tc)
            sched = build_scheduler(opt, steps_per_epoch * n_ep, tc.warmup_ratio)
            if resume is not None and resume["stage"] == s and resume["optimizer"] is not None:
                opt.load_state_dict(resume["optimizer"])
                sched.load_state_dict(resume["scheduler"])
            resume = None
            c = count_parameters(model)
            print(f"stage {s}: backbone trainable fraction={frac} | params total={c['total']:,} "
                  f"trainable={c['trainable']:,} frozen={c['frozen']:,}")

        model.train()
        if model.backbone_frozen:
            model.backbone.eval()
        agg, nb = defaultdict(float), 0
        opt.zero_grad(set_to_none=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with amp_ctx(device, tc.amp_dtype):
                out = model(**model_inputs(batch))
                loss, parts = criterion(out, batch["target"])
            (loss / tc.grad_accum_steps).backward()
            if (i + 1) % tc.grad_accum_steps == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, tc.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
            agg["loss"] += loss.item()
            for k, v in parts.items():
                agg[k] += v
            nb += 1

        arr = collect(model, val_loader, device, criterion)
        row = {"epoch": epoch + 1, "step": state["step"], "stage": s,
               **{f"train_{k}": v / nb for k, v in agg.items()},
               "lr": sched.get_last_lr()[0], "temperature": float(model.temperature.temperature.detach()),
               **quick_metrics(arr)}
        state["history"].append(row)
        state["epoch"] = epoch + 1
        print(f"ep {row['epoch']:>3} | loss {row['train_loss']:.4f} ce {row['train_ce']:.4f} "
              f"brier {row['train_brier']:.4f} conf {row['train_conf']:.4f} lr {row['lr']:.2e} "
              f"T {row['temperature']:.3f} | val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} "
              f"f1 {row['val_f1']:.4f} ece {row['val_ece']:.4f} brier {row['val_brier']:.4f}")

        v = row[tc.monitor]
        if state["best"] is None or (v < state["best"] if lower else v > state["best"]):
            state["best"], state["best_epoch"], state["bad"] = v, row["epoch"], 0
            save_checkpoint(out_dir / "best.pt", model, cfg, labels, state, stage=s)
        else:
            state["bad"] += 1
        save_checkpoint(out_dir / "last.pt", model, cfg, labels, state, opt, sched, s)
        (out_dir / "history.json").write_text(json.dumps(state["history"], indent=2))
        if state["bad"] >= tc.patience:
            print(f"early stopping at epoch {row['epoch']}")
            break

    best_row = next((h for h in state["history"] if h["epoch"] == state["best_epoch"]), None)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"monitor": tc.monitor, "best": state["best"], "best_epoch": state["best_epoch"], "val": best_row}, indent=2))
    return state


def run_training(cfg: Config, train_path, val_path, out_dir, resume: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    labels = load_labels(cfg.data.labels_path)
    if cfg.decision.mode == "fixed" and len(labels) != cfg.model.num_classes:
        raise ValueError(f"labels file has {len(labels)} classes, config has {cfg.model.num_classes}")
    tok_dir = out_dir / "tokenizer"
    if resume:
        tok = build_tokenizer(cfg, load_dir=tok_dir)
    else:
        texts = [t for r in read_jsonl(train_path) for t in record_texts(r)]
        tok = build_tokenizer(cfg, texts=texts)
        tok.save(tok_dir)
    cfg.model.vocab_size = tok.vocab_size
    train_ds = DecisionDataset(train_path, tok, cfg)
    val_ds = DecisionDataset(val_path, tok, cfg)
    collate = make_collate(tok.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)
    model = SystemOneModel(cfg).to(device)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out_dir / "env.json").write_text(json.dumps(env_info(cfg.training.seed, device), indent=2))
    print(f"device: {device} | train {len(train_ds)} | val {len(val_ds)}")
    return train(model, cfg, train_loader, val_loader, device, out_dir, labels,
                 resume_path=out_dir / "last.pt" if resume else None)
