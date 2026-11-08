import random

import numpy as np

from .data import DEFAULT_QUESTION
from .metrics import calculate_ece


def order_report(pred, records: list, n_perms: int = 6, seed: int = 0, batch_size: int = 64) -> dict:
    """Run each question in n_perms option orders (first = original order) and measure order sensitivity."""
    rng = random.Random(seed)
    records = [r for r in records if len(set(r["options"])) == len(r["options"])]
    reqs, meta = [], []
    for ri, r in enumerate(records):
        opts = list(r["options"])
        perms = [list(range(len(opts)))]
        for _ in range(n_perms - 1):
            p = list(range(len(opts)))
            rng.shuffle(p)
            perms.append(p)
        for p in perms:
            reqs.append({"context": r.get("context", ""), "question": r.get("question", DEFAULT_QUESTION),
                         "options": [opts[i] for i in p]})
            meta.append((ri, p))
    res = pred.predict_batch(reqs, batch_size)

    per, pos_stats, confs, corrects = {}, {}, [], []
    for out, (ri, p) in zip(res, meta):
        r = records[ri]
        opts, ti = r["options"], int(r["target"])
        pred_idx = opts.index(out["prediction"])
        aligned = np.array([out["scores"][o] for o in opts])
        correct = float(pred_idx == ti)
        pos = p.index(ti)
        per.setdefault(ri, []).append({"pred": pred_idx, "conf": out["confidence"], "aligned": aligned, "ptrue": aligned[ti]})
        s = pos_stats.setdefault(len(opts), {}).setdefault(pos, [])
        s.append(correct)
        confs.append(out["confidence"])
        corrects.append(correct)

    flips, pair_flips, conf_std, ptrue_std, avg_ok = [], [], [], [], []
    for ri, runs in per.items():
        preds = [x["pred"] for x in runs]
        flips.append(len(set(preds)) > 1)
        pair_flips += [x != preds[0] for x in preds[1:]]
        conf_std.append(np.std([x["conf"] for x in runs]))
        ptrue_std.append(np.std([x["ptrue"] for x in runs]))
        avg_ok.append(int(np.mean([x["aligned"] for x in runs], axis=0).argmax() == int(records[ri]["target"])))

    position_acc = {str(k): [float(np.mean(v[p])) if p in v else None for p in range(k)] for k, v in pos_stats.items()}
    gaps = [max(a) - min(a) for a in ([x for x in v if x is not None] for v in position_acc.values()) if len(a) > 1]
    ece, _ = calculate_ece(np.array(confs), np.array(corrects))
    return {
        "questions": len(per), "permutations": n_perms,
        "accuracy": float(np.mean(corrects)),
        "flip_rate": float(np.mean(flips)),
        "pairwise_flip_rate": float(np.mean(pair_flips)) if pair_flips else 0.0,
        "position_accuracy": position_acc,
        "position_gap": float(max(gaps)) if gaps else 0.0,
        "worst_position_accuracy": float(min(x for v in position_acc.values() for x in v if x is not None)),
        "confidence_std": float(np.mean(conf_std)),
        "true_option_prob_std": float(np.mean(ptrue_std)),
        "ece": float(ece),
        "perm_average_accuracy": float(np.mean(avg_ok)),
    }


def leak_test(pred, records: list, limit: int = 50) -> dict:
    """Change the text of the FIRST option and see how much the LAST option's raw score moves (0 = options are isolated)."""
    diffs = []
    for r in records[:limit]:
        if len(r["options"]) < 2:
            continue
        ctx, q = r.get("context", ""), r.get("question", DEFAULT_QUESTION)

        def score0(opts):
            batch, _ = pred.encode(ctx, opts, q)
            return float(pred.forward(batch).logits[0, len(opts) - 1])

        base = score0(list(r["options"]))
        changed = list(r["options"])
        changed[0] = "completely different unrelated words here"
        diffs.append(abs(score0(changed) - base))
    return {"mean_score_change": float(np.mean(diffs)), "max_score_change": float(np.max(diffs))}
