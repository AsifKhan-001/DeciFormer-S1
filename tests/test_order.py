import json
import random

import numpy as np
import pytest
import torch

from system_one.config import Config
from system_one.data import DecisionDataset, encode_dynamic, make_collate
from system_one.losses import TotalLoss
from system_one.model.system_one import SystemOneModel
from system_one.predictor import Predictor
from system_one.robustness import leak_test, order_report
from system_one.tokenizer import SimpleTokenizer
from system_one.trainer import model_inputs, run_training

OPTS = ["technical team", "finance team", "account team"]
WORDS = {0: "crash error bug freeze", 1: "invoice refund bill charge", 2: "password login locked profile"}


def cfg_for(isolate, **train):
    return Config.from_dict({
        "model": {"vocab_size": 0, "max_seq_len": 64, "hidden_size": 32, "num_layers": 2, "num_heads": 4,
                  "intermediate_size": 64, "num_classes": 3, "dropout": 0.0},
        "decision": {"mode": "dynamic", "pooling": "last", "max_options": 6, "isolate_options": isolate},
        "calibration": {"hidden_size": 16}, "training": train})


def tok_for(rows):
    return SimpleTokenizer.build([t for r in rows for t in [r["context"], *r["options"]]], 6)


def make_rows(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        y = rng.randrange(3)
        opts = OPTS[:]
        rng.shuffle(opts)
        rows.append({"context": " ".join(rng.choices(WORDS[y].split(), k=3)) + " help", "question": "Which team?",
                     "options": opts, "target": opts.index(OPTS[y])})
    return rows


def batch_for(tok, ctx, options, shared=False, ml=64):
    enc = encode_dynamic(tok, ctx, "Which team?", options, ml, 6, shared)
    return make_collate(tok.pad_id)([{**enc, "target": 0}])


@pytest.mark.parametrize("isolate", [True, False])
def test_isolation_and_invariance(isolate):
    torch.manual_seed(0)
    rows = make_rows(20)
    tok = tok_for(rows)
    cfg = cfg_for(isolate)
    cfg.model.vocab_size = tok.vocab_size
    model = SystemOneModel(cfg).eval()
    a = model(**model_inputs(batch_for(tok, "crash error help", OPTS, isolate)))
    changed = ["invoice bill refund words"] + OPTS[1:]
    b = model(**model_inputs(batch_for(tok, "crash error help", changed, isolate)))
    leak = abs(float(a.logits[0, -1].detach()) - float(b.logits[0, -1].detach()))
    perm = [2, 0, 1]
    c = model(**model_inputs(batch_for(tok, "crash error help", [OPTS[i] for i in perm], isolate)))
    aligned = torch.zeros(3).scatter(0, torch.tensor(perm), c.probabilities[0])
    diff = float((aligned - a.probabilities[0]).abs().max().detach())
    if isolate:
        assert leak < 1e-5 and diff < 1e-5
    else:
        assert leak > 1e-5 and diff > 1e-5


@pytest.mark.parametrize("isolate", [True, False])
def test_pair_batch_and_order_loss(isolate):
    torch.manual_seed(0)
    rows = make_rows(8)
    tok = tok_for(rows)
    cfg = cfg_for(isolate, order_mode="consistency")
    cfg.model.vocab_size = tok.vocab_size
    import tempfile
    path = tempfile.mktemp(suffix=".jsonl")
    open(path, "w").write("\n".join(json.dumps(r) for r in rows))
    ds = DecisionDataset(path, tok, cfg, pair=True)
    batch = make_collate(tok.pad_id)([ds[i] for i in range(4)])
    assert batch["input_ids"].size(0) == 8 and batch["perm"].shape == (8, 3)
    model = SystemOneModel(cfg).eval()
    out = model(**model_inputs(batch))
    loss, parts = TotalLoss(0.5, 0.5, 1.0)(out, batch["target"], batch["perm"])
    assert (parts["order"] < 1e-6) if isolate else (parts["order"] > 1e-6)
    loss.backward()


def test_training_and_order_report(tmp_path):
    train, val = make_rows(90, 1), make_rows(45, 2)
    for name, rows in (("train", train), ("val", val)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "labels.json").write_text(json.dumps({"0": "a", "1": "b", "2": "c"}))
    cfg = cfg_for(True, order_mode="consistency", epochs=25, batch_size=15, lr=0.005, backbone_lr=0.005, patience=99)
    cfg.data.labels_path = str(tmp_path / "labels.json")
    state = run_training(cfg, tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "exp")
    assert state["history"][-1]["val_acc"] >= 0.9
    pred = Predictor(tmp_path / "exp" / "best.pt", device="cpu")
    rep = order_report(pred, val, n_perms=4)
    assert rep["flip_rate"] <= 0.02 and rep["position_gap"] <= 0.1
    assert leak_test(pred, val)["max_score_change"] < 1e-4
