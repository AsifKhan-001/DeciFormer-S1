import copy, json, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ENC = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
D, K, SEEDS = "data/metatool", 24, 3
data = {s: [json.loads(l) for l in open(f"{D}/{s}.jsonl")] for s in ("train", "val", "test")}

pool = sorted({t for r in data["test"] for t in r["options"]})
rng = random.Random(1)
many = []
for r in data["test"]:
    g = r["options"][r["target"]]
    o = [g] + rng.sample([t for t in pool if t != g], 19)
    rng.shuffle(o)
    many.append(dict(r, options=o, target=o.index(g)))

texts = sorted({t for rs in (*data.values(), many) for r in rs for t in [r["context"], *r["options"]]})
enc = SentenceTransformer(ENC, device="mps" if torch.backends.mps.is_available() else "cpu")
E = enc.encode(texts, normalize_embeddings=True, batch_size=64, convert_to_tensor=True).cpu()
ix = {t: i for i, t in enumerate(texts)}


def pack(rows):
    q = E[[ix[r["context"]] for r in rows]]
    o = torch.zeros(len(rows), K, E.size(1))
    m = torch.zeros(len(rows), K, dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["options"])
        o[i, :n] = E[[ix[t] for t in r["options"]]]
        m[i, :n] = True
    return q, o, m, torch.tensor([r["target"] for r in rows])


P = {"train": pack(data["train"]), "val": pack(data["val"]), "test": pack(data["test"]), "many": pack(many)}


def feats(s, m):
    sm = s.masked_fill(~m, -1e4)
    cnt = m.sum(1, keepdim=True)
    mean = (s * m).sum(1, keepdim=True) / cnt
    var = (((s - mean) ** 2) * m).sum(1, keepdim=True) / cnt
    top = sm.topk(2, dim=1).values
    other = torch.where(sm == top[:, :1], top[:, 1:2], top[:, :1])
    return torch.stack([s, s - mean, s - other, (s - mean) / (var.sqrt() + 1e-4)], -1)


class Head(nn.Module):
    def __init__(self, d, adapt, setl, r=64):
        super().__init__()
        self.adapt, self.setl = adapt, setl
        if adapt:
            self.aq, self.bq = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
            self.ao, self.bo = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
            nn.init.zeros_(self.bq.weight)
            nn.init.zeros_(self.bo.weight)
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.inp = nn.Linear(4, 32)
        self.att = nn.MultiheadAttention(32, 4, batch_first=True)
        self.out = nn.Linear(32, 1)
        nn.init.zeros_(self.out.weight)

    def forward(self, q, o, m):
        if self.adapt:
            q = F.normalize(q + self.bq(self.aq(q)), dim=-1)
            o = F.normalize(o + self.bo(self.ao(o)), dim=-1)
        s = torch.einsum("bd,bkd->bk", q, o)
        logit = self.scale * s
        if self.setl:
            h = F.gelu(self.inp(feats(s, m)))
            h = h + self.att(h, h, h, key_padding_mask=~m, need_weights=False)[0]
            logit = logit + self.out(h).squeeze(-1)
        return logit.masked_fill(~m, -1e9)


def train(model, seed):
    tq, to, tm, ty = P["train"]
    vq, vo, vm, vy = P["val"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = 1e9, None, 0
    for _ in range(80):
        model.train()
        for b in torch.randperm(len(ty), generator=g).split(64):
            loss = F.cross_entropy(model(tq[b], to[b], tm[b]), ty[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(vq, vo, vm), vy).item()
        if vl < best:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(state)


def fit_T(logits, y):
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(logits / t.exp(), y)
        l.backward()
        return l

    opt.step(closure)
    return t.exp().item()


def ece(p, y, bins=10):
    conf, pred = p.max(1)
    acc = (pred == y).float()
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        mk = (conf > lo) & (conf <= lo + 1 / bins)
        if mk.any():
            e += mk.float().mean().item() * abs(conf[mk].mean().item() - acc[mk].mean().item())
    return e


def sel(p, y, fr=(1.0, 0.8, 0.6, 0.4)):
    conf, pred = p.max(1)
    a = (pred == y).float()[conf.argsort(descending=True)]
    return [a[: int(len(a) * f)].mean().item() for f in fr]


def pred_names(model, rows):
    q, o, m, _ = pack(rows)
    with torch.no_grad():
        p = model(q, o, m).argmax(1).tolist()
    return [r["options"][i] for r, i in zip(rows, p)]


def flip(model, rows, n=10):
    base = pred_names(model, rows)
    ch = np.zeros(len(rows), bool)
    for _ in range(n):
        sh = [dict(r, options=random.sample(r["options"], len(r["options"]))) for r in rows]
        ch |= np.array([a != b for a, b in zip(pred_names(model, sh), base)])
    return float(ch.mean())


def run(variant, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = Head(E.size(1), "adapt" in variant, "set" in variant)
    if variant != "sim":
        train(model, seed)
    model.eval()
    with torch.no_grad():
        T = fit_T(model(*P["val"][:3]), P["val"][3])
        q, o, m, y = P["test"]
        p = (model(q, o, m) / T).softmax(-1)
        qm, om, mm, ym = P["many"]
        acc20 = (model(qm, om, mm).argmax(1) == ym).float().mean().item()
    c = sel(p, y)
    return dict(acc=(p.argmax(1) == y).float().mean().item(), ece=ece(p, y), flip=flip(model, data["test"]),
                acc20=acc20, cov80=c[1], cov60=c[2], cov40=c[3])


res = {}
for v in ("sim", "adapt", "set", "adapt+set"):
    rs = [run(v, s) for s in range(SEEDS)]
    res[v] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
    print(v.ljust(10), " ".join(f"{k} {a:.3f}±{s:.3f}" for k, (a, s) in res[v].items()))
print("test n", len(data["test"]), "| chance 2-6 opts ~0.29 | chance 20 opts 0.05")
json.dump(res, open("experiments/two_tower_results.json", "w"), indent=1)