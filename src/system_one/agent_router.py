import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SCALE = 20.0
Tools = Union[Dict[str, str], Sequence[Union[str, Tuple[str, str]]]]


@dataclass
class Decision:
    id: int
    tool: str
    confidence: float
    escalate: bool
    reason: str
    shadow: bool
    scores: Dict[str, float]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _tools(tools: Tools) -> Tuple[List[str], List[str]]:
    if isinstance(tools, dict):
        items = list(tools.items())
    else:
        items = [(t, "") if isinstance(t, str) else (t[0], t[1]) for t in tools]
    if not items:
        raise ValueError("no tools given")
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    return names, [f"{n}: {d}" if d else n for n, d in items]


def pick_tau(conf: np.ndarray, ok: np.ndarray, target: float, min_n: int = 30) -> float:
    if len(conf) < min_n:
        return 1.01
    o = np.argsort(-conf)
    c, k = conf[o], ok[o]
    err = np.cumsum(1 - k) / np.arange(1, len(c) + 1)
    good = np.where(err <= target)[0]
    good = good[good >= min_n - 1]
    return float(c[good[-1]]) if len(good) else 1.01


class ToolRouter:
    def __init__(self, encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
                 encode_fn: Optional[Callable] = None, target_error: float = 0.05, window: int = 1000,
                 min_fail: int = 2, fail_rate: float = 0.4, explore: float = 0.1, seed: int = 0):
        if encode_fn is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(encoder)
            encode_fn = lambda xs: model.encode(list(xs), normalize_embeddings=True)
        self._enc = encode_fn
        self._cache: Dict[str, np.ndarray] = {}
        self._qcache: Dict[str, np.ndarray] = {}
        self.target_error, self.window = target_error, window
        self.min_fail, self.fail_rate, self.explore = min_fail, fail_rate, explore
        self.T, self.tau, self.min_sim = 1.0, 0.0, None
        self.t = 0
        self.hist: Dict[str, List[List[float]]] = {}
        self.pending: Dict[int, List] = {}
        self.rng = random.Random(seed)

    def _emb(self, texts: Sequence[str]) -> np.ndarray:
        miss = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if miss:
            for t, v in zip(miss, _unit(np.asarray(self._enc(miss), dtype=np.float64))):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def _q(self, task: str) -> np.ndarray:
        if task not in self._qcache:
            if len(self._qcache) > 5000:
                self._qcache.clear()
            self._qcache[task] = _unit(np.asarray(self._enc([task]), dtype=np.float64))[0]
        return self._qcache[task]

    def _sims(self, task: str, texts: Sequence[str]) -> np.ndarray:
        return self._emb(texts) @ self._q(task)

    def _probs(self, sims: np.ndarray) -> np.ndarray:
        return _softmax(SCALE * sims / self.T)

    def _flagged(self, tool: str) -> bool:
        h = [x for x in self.hist.get(tool, []) if x[0] > self.t - self.window]
        f = sum(1 - x[1] for x in h)
        return f >= self.min_fail and f / len(h) > self.fail_rate

    def route(self, task: str, tools: Tools) -> Decision:
        names, texts = _tools(tools)
        sims = self._sims(task, texts)
        p = self._probs(sims)
        i = int(p.argmax())
        self.t += 1
        escalate, reason, shadow = False, "ok", False
        if self.min_sim is not None and sims[i] < self.min_sim:
            escalate, reason = True, "no tool fits"
        elif p[i] < self.tau:
            escalate, reason = True, "low confidence"
        elif self._flagged(names[i]):
            escalate, reason = True, "tool recently failing"
            shadow = self.rng.random() < self.explore
        self.pending[self.t] = [names[i], self.t]
        if len(self.pending) > 5000:
            del self.pending[min(self.pending)]
        return Decision(self.t, names[i], float(p[i]), escalate, reason, shadow,
                        {n: float(x) for n, x in zip(names, p)})

    def report(self, decision_id: int, success: bool) -> None:
        tool, t = self.pending.pop(decision_id)
        h = self.hist.setdefault(tool, [])
        h.append([t, 1.0 if success else 0.0])
        self.hist[tool] = [x for x in h if x[0] > self.t - self.window]

    def calibrate(self, rows: Sequence[dict], target_error: Optional[float] = None) -> dict:
        from scipy.optimize import minimize_scalar
        target = self.target_error if target_error is None else target_error
        sims = [self._sims(r["context"], _tools(r["options"])[1]) for r in rows]
        y = np.array([r["target"] for r in rows])

        def nll(log_t):
            tot = 0.0
            for s, k in zip(sims, y):
                z = SCALE * s / np.exp(log_t)
                z = z - z.max()
                tot += np.log(np.exp(z).sum()) - z[k]
            return tot / len(y)

        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        conf, ok = self._conf_ok(sims, y)
        self.target_error = target
        self.tau = pick_tau(conf, ok, target)
        act = conf >= self.tau
        return {"n": len(rows), "T": self.T, "tau": self.tau, "accuracy": float(ok.mean()),
                "coverage": float(act.mean()), "acted_error": float(1 - ok[act].mean()) if act.any() else None}

    def _conf_ok(self, sims: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ps = [self._probs(s) for s in sims]
        return np.array([p.max() for p in ps]), np.array([float(p.argmax() == k) for p, k in zip(ps, y)])

    def evaluate(self, rows: Sequence[dict], perms: int = 10, seed: int = 0) -> dict:
        sims = [self._sims(r["context"], _tools(r["options"])[1]) for r in rows]
        y = np.array([r["target"] for r in rows])
        conf, ok = self._conf_ok(sims, y)
        act = conf >= self.tau
        rng, flips = random.Random(seed), 0
        for r in rows:
            base = r["options"][int(self._sims(r["context"], _tools(r["options"])[1]).argmax())]
            for _ in range(perms):
                o = list(r["options"])
                rng.shuffle(o)
                if o[int(self._sims(r["context"], _tools(o)[1]).argmax())] != base:
                    flips += 1
                    break
        srt = ok[np.argsort(-conf)]
        out = {"n": len(rows), "accuracy": float(ok.mean()), "coverage": float(act.mean()),
               "acted_error": float(1 - ok[act].mean()) if act.any() else None,
               "flip_rate": flips / len(rows), "target_error": self.target_error}
        for f in (0.8, 0.6, 0.4):
            out[f"acc_top{int(f * 100)}pct_confident"] = float(srt[: max(1, int(len(srt) * f))].mean())
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        s = {"T": self.T, "tau": self.tau, "min_sim": self.min_sim, "t": self.t, "hist": self.hist,
             "pending": {str(k): v for k, v in self.pending.items()}, "target_error": self.target_error}
        with open(path + ".tmp", "w") as f:
            json.dump(s, f)
        os.replace(path + ".tmp", path)

    def load(self, path: str) -> None:
        s = json.load(open(path))
        self.T, self.tau, self.min_sim, self.t = s["T"], s["tau"], s["min_sim"], s["t"]
        self.hist = s["hist"]
        self.pending = {int(k): v for k, v in s["pending"].items()}
        self.target_error = s.get("target_error", self.target_error)

    def status(self) -> dict:
        bad = [t for t in self.hist if self._flagged(t)]
        return {"decisions": self.t, "T": self.T, "tau": self.tau, "min_sim": self.min_sim, "flagged_tools": bad,
                "tools_with_feedback": len(self.hist)}
