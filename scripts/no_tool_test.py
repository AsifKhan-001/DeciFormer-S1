import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.agent_router import ToolRouter

BASE = "data/raw/MetaTool/dataset"
des = json.load(open(f"{BASE}/plugin_des.json"))
data = json.load(open(f"{BASE}/tmp_dataset/Task1.json"))
random.seed(0)
names = sorted(des)


def label(n):
    d = des[n] if isinstance(des[n], str) else " ".join(des[n])
    return f"{n}: {' '.join(d.split()[:12])}"


rows = []
for r in data:
    if not r.get("query"):
        continue
    gold = str(r.get("tool") or "None").strip()
    if gold == "None":
        opts = random.sample(names, 6)
        rows.append((r["query"], [label(n) for n in opts], None))
    elif gold in des:
        opts = [gold] + random.sample([n for n in names if n != gold], 5)
        random.shuffle(opts)
        rows.append((r["query"], [label(n) for n in opts], opts.index(gold)))
random.shuffle(rows)
half = len(rows) // 2
router = ToolRouter()
router.load("experiments/router_state.json")


def feats(part):
    out = []
    for q, opts, g in part:
        s = router._sims(q, opts)
        p = router._probs(s)
        out.append((float(p.max()), float(s.max()), g is not None, g is not None and int(p.argmax()) == g))
    return np.array(out, dtype=float)


cal, tst = feats(rows[:half]), feats(rows[half:])
min_sim = float(np.percentile(cal[cal[:, 2] == 1, 1], 30))
pos, neg = tst[tst[:, 2] == 1], tst[tst[:, 2] == 0]
print(f"test questions: {len(pos):.0f} need a tool, {len(neg):.0f} need no tool")
print(f"AUROC separating 'tool needed' from 'no tool needed': softmax confidence {roc_auc_score(tst[:, 2], tst[:, 0]):.3f} | best cosine similarity {roc_auc_score(tst[:, 2], tst[:, 1]):.3f}")
for name, use_sim in (("tau only (current router)", False), (f"tau + min_sim {min_sim:.3f}", True)):
    act = lambda a: (a[:, 0] >= router.tau) & ((a[:, 1] >= min_sim) if use_sim else True)
    ap, an = act(pos), act(neg)
    err = 1 - pos[ap, 3].mean() if ap.any() else float("nan")
    print(f"{name}: answers {ap.mean():.3f} of tool-needed questions (error among them {err:.3f}) | wrongly answers {an.mean():.3f} of no-tool questions")
