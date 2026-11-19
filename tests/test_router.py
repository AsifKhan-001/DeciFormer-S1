import random
import zlib

import numpy as np
import pytest

from system_one.agent_router import ToolRouter


def enc(texts):
    out = []
    for t in texts:
        v = np.zeros(64)
        for w in t.lower().replace(":", " ").split():
            v += np.random.default_rng(zlib.crc32(w.encode())).normal(size=64)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(out)


TOOLS = {"calculator": "add numbers math", "weather": "forecast rain city", "calendar": "meeting schedule date",
         "translator": "translate language text"}


def make():
    return ToolRouter(encode_fn=enc)


def test_picks_right_tool():
    d = make().route("what is the weather forecast in rain", TOOLS)
    assert d.tool == "weather" and not d.escalate


def test_order_invariance():
    r = make()
    items = list(TOOLS.items())
    base = r.route("schedule a meeting date", dict(items)).scores
    for s in range(5):
        random.Random(s).shuffle(items)
        d = r.route("schedule a meeting date", dict(items))
        assert d.tool == "calendar"
        assert all(abs(d.scores[k] - base[k]) < 1e-9 for k in base)


def test_calibrate_and_evaluate():
    rows = []
    names = list(TOOLS)
    rng = random.Random(0)
    for _ in range(120):
        n = rng.choice(names)
        opts = [f"{k}: {v}" for k, v in TOOLS.items()]
        rng.shuffle(opts)
        rows.append({"context": TOOLS[n], "options": opts, "target": opts.index(f"{n}: {TOOLS[n]}")})
    r = make()
    out = r.calibrate(rows, 0.05)
    assert out["accuracy"] > 0.95 and out["coverage"] > 0.5
    ev = r.evaluate(rows)
    assert ev["flip_rate"] == 0.0


def test_tracker_flags_and_recovers():
    r = make()
    r.tau, r.explore = 0.0, 0.0
    for _ in range(3):
        d = r.route("add numbers math", TOOLS)
        r.report(d.id, False)
    d = r.route("add numbers math", TOOLS)
    assert d.escalate and d.reason == "tool recently failing"
    r.t += 2000
    assert not r.route("add numbers math", TOOLS).escalate


def test_shadow_and_save_load(tmp_path):
    r = make()
    r.tau, r.explore = 0.0, 1.0
    for _ in range(3):
        r.report(r.route("add numbers math", TOOLS).id, False)
    d = r.route("add numbers math", TOOLS)
    assert d.escalate and d.shadow
    p = str(tmp_path / "s.json")
    r.save(p)
    r2 = make()
    r2.load(p)
    assert r2.t == r.t and r2.route("add numbers math", TOOLS).escalate


def test_unknown_report_and_low_confidence():
    r = make()
    with pytest.raises(KeyError):
        r.report(999, True)
    r.tau = 1.01
    d = r.route("add numbers math", TOOLS)
    assert d.escalate and d.reason == "low confidence"
