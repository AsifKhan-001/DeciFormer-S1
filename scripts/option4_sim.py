import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

ENC = "sentence-transformers/all-MiniLM-L6-v2"
SMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MODE = sys.argv[2] if len(sys.argv) > 2 else "noise"
TARGET, EPS, STEPS, WARM, UPD, WIN, K, SCALE, SEEDS, BLK = 0.05, 0.1, 6000, 500, 250, 1500, 20, 20.0, 5, 500
METHODS = ("fixed", "naive", "explore", "pertool", "oracle")
TWIN, MINFAIL, FAILRATE = 1000, 2, 0.4
B1, B2 = WARM + int(0.25 * (STEPS - WARM)), WARM + int(0.75 * (STEPS - WARM))

rows = [json.loads(l) for s in ("train", "val", "test") for l in open(f"data/metatool/{s}.jsonl")]
gold = [r["options"][r["target"]] for r in rows]
pool = sorted(set(gold))
pid = {t: i for i, t in enumerate(pool)}
gidx = np.array([pid[g] for g in gold])
assert len(pool) >= K, "not enough distinct tools"
enc = SentenceTransformer(ENC)
qE = enc.encode([r["context"] for r in rows], normalize_embeddings=True)
oE = enc.encode(pool, normalize_embeddings=True)
D, NP = qE.shape[1], len(pool)
S = oE @ oE.T
np.fill_diagonal(S, -1)
sib = S.argmax(1)
broken = np.random.default_rng(7).random(NP) < 0.3


def stream(rng):
    t = np.arange(STEPS)
    ph = np.sin(np.pi * np.clip((t - WARM) / (STEPS - WARM), 0, 1)) ** 2
    conf, ok, ch = np.zeros(STEPS), np.zeros(STEPS), np.zeros(STEPS, int)
    for k in range(STEPS):
        i = rng.integers(len(rows))
        g = gidx[i]
        right, sig = 0, SMAX * ph[k]
        if MODE == "concept":
            others = np.setdiff1d(np.arange(NP), [g, sib[g]])
            opts = np.concatenate([[g, sib[g]], rng.permutation(others)[:K - 2]])
            right, sig = int(ph[k] > 0.5 and broken[g]), 0.0
        else:
            opts = np.concatenate([[g], rng.permutation(np.delete(np.arange(NP), g))[:K - 1]])
        q = qE[i] + rng.normal(0, sig / np.sqrt(D), D)
        q /= np.linalg.norm(q)
        z = SCALE * (oE[opts] @ q)
        p = np.exp(z - z.max())
        p /= p.sum()
        conf[k], ok[k], ch[k] = p.max(), float(p.argmax() == right), opts[p.argmax()]
    return conf, ok, ch


def pick_tau(c, ok, w):
    if len(c) < 30:
        return None
    o = np.argsort(-c)
    c, ok, w = c[o], ok[o], w[o]
    err = np.cumsum(w * (1 - ok)) / np.cumsum(w)
    good = np.where(err <= TARGET)[0]
    good = good[good >= 29]
    return c[good[-1]] if len(good) else 1.01


def simulate(method, conf, ok, ch, rng):
    tau = pick_tau(conf[:WARM], ok[:WARM], np.ones(WARM))
    log, hist, acted = [], {}, np.zeros(STEPS, bool)
    for t in range(WARM, STEPS):
        a = conf[t] >= tau
        if method == "pertool":
            h = [x for x in hist.get(ch[t], []) if x[0] > t - TWIN]
            f = sum(1 - x[1] for x in h)
            bad = f >= MINFAIL and f / len(h) > FAILRATE
            acted[t] = a and not bad
            if a and (not bad or rng.random() < EPS):
                hist.setdefault(ch[t], []).append((t, ok[t]))
            continue
        acted[t] = a
        if a:
            log.append((t, conf[t], ok[t], 1.0))
        elif method == "explore" and rng.random() < EPS:
            log.append((t, conf[t], ok[t], 1 / EPS))
        if method != "fixed" and (t - WARM + 1) % UPD == 0:
            if method == "oracle":
                lo = max(0, t - WIN + 1)
                c, o, w = conf[lo:t + 1], ok[lo:t + 1], np.ones(t + 1 - lo)
            else:
                L = np.array([x for x in log if x[0] > t - WIN]).reshape(-1, 4)
                c, o, w = L[:, 1], L[:, 2], L[:, 3]
            new = pick_tau(c, o, w)
            if new is not None:
                tau = new
    return acted


def metrics(acted, ok):
    out = []
    for lo, hi in ((WARM, B1), (B1, B2), (B2, STEPS)):
        a, o = acted[lo:hi], ok[lo:hi]
        out += [1 - o[a].mean() if a.any() else 0.0, a.mean()]
    return out


res, accs = {m: [] for m in METHODS}, []
for seed in range(SEEDS):
    conf, ok, ch = stream(np.random.default_rng(seed))
    accs.append((ok[:WARM].mean(), min(ok[b:b + BLK].mean() for b in range(WARM, STEPS, BLK))))
    for m in METHODS:
        res[m].append(metrics(simulate(m, conf, ok, ch, np.random.default_rng(100 + seed)), ok))
print("router top-1 acc (20 options): clean %.3f | worst block %.3f" % tuple(np.mean(accs, 0)))
print(f"target acted-error {TARGET} | mode {MODE} | drift strength {SMAX} | seeds {SEEDS}")
print("method    BEFORE err/cov     DURING err/cov     AFTER err/cov")
for m in METHODS:
    a = np.mean(res[m], 0)
    print(m.ljust(8), "  ".join(f"{a[i]:.3f} / {a[i + 1]:.3f}" for i in (0, 2, 4)))
