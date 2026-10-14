import json

import numpy as np
import pytest
import torch

from system_one.config import Config
from system_one.data import DecisionDataset, make_collate
from system_one.losses import BrierLoss, TotalLoss
from system_one.metrics import calculate_ece
from system_one.model.heads import CalibrationHead, DecisionHead
from system_one.model.layers import CausalSelfAttention, RMSNorm, RotaryEmbedding, SwiGLU
from system_one.model.system_one import SystemOneModel
from system_one.policy import DecisionPolicy
from system_one.predictor import Predictor
from system_one.tokenizer import SimpleTokenizer
from system_one.trainer import run_training


def make_cfg(mode="fixed", pooling="last", k=5, **train):
    return Config.from_dict({
        "model": {"vocab_size": 60, "max_seq_len": 32, "hidden_size": 32, "num_layers": 2, "num_heads": 4,
                  "intermediate_size": 64, "num_classes": k},
        "decision": {"mode": mode, "pooling": pooling, "max_options": 6},
        "calibration": {"hidden_size": 16}, "training": train})


def batch(b=3, n=10, v=60):
    ids = torch.randint(3, v, (b, n))
    mask = torch.ones(b, n, dtype=torch.long)
    mask[1, 7:] = 0
    return ids, mask


def test_rmsnorm():
    y = RMSNorm(16)(torch.randn(4, 5, 16) * 7)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 5), atol=1e-3)


def test_rope():
    rope = RotaryEmbedding(8, 16)
    q, k = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    q2, k2 = rope(q, k)
    assert q2.shape == q.shape and torch.allclose(q2[:, :, 0], q[:, :, 0])
    assert not torch.allclose(q2[:, :, 3], q[:, :, 3])
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_attention_causal_and_padding():
    torch.manual_seed(0)
    attn = CausalSelfAttention(32, 4, 16, 10000.0, 0.0).eval()
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 5] += 5
    assert torch.allclose(attn(x)[:, :5], attn(x2)[:, :5], atol=1e-5)
    short = attn(x[:, :4])
    padded = torch.cat([x[:, :4], torch.randn(1, 2, 32)], 1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    assert torch.allclose(short, attn(padded, mask)[:, :4], atol=1e-5)


def test_swiglu_shape():
    assert SwiGLU(32, 64)(torch.randn(2, 5, 32)).shape == (2, 5, 32)


@pytest.mark.parametrize("k", [2, 5, 10])
def test_decision_head(k):
    assert DecisionHead(32, k)(torch.randn(4, 32)).shape == (4, k)


def test_calibration_head():
    c = CalibrationHead(32, 16)(torch.randn(4, 32), torch.rand(4, 1))
    assert c.shape == (4, 1) and (c >= 0).all() and (c <= 1).all()


@pytest.mark.parametrize("pooling", ["last", "decision_token", "mean", "attention"])
def test_fixed_forward(pooling):
    model = SystemOneModel(make_cfg(pooling=pooling)).eval()
    ids, mask = batch()
    out = model(ids, mask)
    assert out.logits.shape == out.probabilities.shape == (3, 5)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(out.logits).all() and out.confidence.shape == (3, 1)
    assert (out.prediction < 5).all() and (out.entropy >= 0).all() and (out.entropy <= 1 + 1e-5).all()


def test_gradient_flow_and_freeze():
    model = SystemOneModel(make_cfg())
    ids, mask = batch()
    out = model(ids, mask)
    TotalLoss()(out, torch.tensor([0, 1, 2]))[0].backward()
    for mod in (model.backbone.embed, model.backbone.blocks[0].attention, model.backbone.blocks[0].ffn,
                model.decision_head, model.calibration_head):
        assert all(p.grad is not None for p in mod.parameters())
    assert model.temperature.raw.grad is not None
    model.zero_grad()
    model.set_backbone_trainable(0.0)
    out = model(ids, mask)
    TotalLoss()(out, torch.tensor([0, 1, 2]))[0].backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.decision_head.parameters())


def test_losses():
    probs = torch.eye(3)
    assert BrierLoss()(probs, torch.tensor([0, 1, 2])) == 0
    conf, _ = calculate_ece(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert conf == 0


def test_dynamic_forward():
    model = SystemOneModel(make_cfg(mode="dynamic")).eval()
    ids, mask = batch()
    idx = torch.tensor([[1, 3, 5], [2, 4, 0], [1, 2, 3]])
    om = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    out = model(ids, mask, option_indices=idx, option_mask=om)
    assert out.probabilities.shape == (3, 3)
    assert out.probabilities[1, 2] == 0 and torch.allclose(out.probabilities.sum(-1), torch.ones(3), atol=1e-5)
    TotalLoss()(model(ids, mask, option_indices=idx, option_mask=om), torch.tensor([0, 1, 2]))[0].backward()


def test_policy():
    pol = DecisionPolicy(0.85, 0.5, 0.15)
    assert pol([0.95, 0.7, 0.2, 0.95], [0.5, 0.5, 0.5, 0.05]) == ["ROUTE", "CLARIFY", "ABSTAIN", "CLARIFY"]


def test_config_validation():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"hidden_size": 30, "num_heads": 4}})


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    words = {0: "crash error bug", 1: "invoice refund bill", 2: "password login account", 3: "advert campaign banner"}
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(80):
        y = int(rng.integers(0, 4))
        w = words[y].split()
        rows.append({"query": " ".join(rng.choice(w, 3)) + " help", "label": y})
    for name in ("train", "val"):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (root / "labels.json").write_text(json.dumps({str(i): f"c{i}" for i in range(4)}))
    cfg = Config.from_dict({
        "model": {"max_seq_len": 16, "hidden_size": 32, "num_layers": 2, "num_heads": 4,
                  "intermediate_size": 64, "num_classes": 4, "dropout": 0.0},
        "calibration": {"hidden_size": 16}, "data": {"labels_path": str(root / "labels.json")},
        "training": {"epochs": 30, "batch_size": 20, "lr": 0.005, "backbone_lr": 0.005, "patience": 99}})
    state = run_training(cfg, root / "train.jsonl", root / "val.jsonl", root / "exp")
    return root, state


def test_overfit_and_inference(tiny_run):
    root, state = tiny_run
    assert state["history"][-1]["val_acc"] >= 0.95
    assert state["history"][-1]["train_loss"] < state["history"][0]["train_loss"]
    pred = Predictor(root / "exp" / "best.pt", device="cpu")
    res = pred.predict("refund invoice bill help")
    assert res["prediction"] == "c1" and res["action"] in {"ROUTE", "CLARIFY", "ABSTAIN"}
    assert abs(sum(res["scores"].values()) - 1) < 1e-4 and 0 <= res["confidence"] <= 1


def test_resume(tiny_run):
    root, _ = tiny_run
    import system_one.trainer as tr
    cfg = Config.from_dict(tr.load_checkpoint(root / "exp" / "last.pt")["config"])
    cfg.training.epochs = 32
    state = run_training(cfg, root / "train.jsonl", root / "val.jsonl", root / "exp", resume=True)
    assert state["epoch"] == 32 and len(state["history"]) == 32
