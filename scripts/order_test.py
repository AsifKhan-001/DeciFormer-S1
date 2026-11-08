import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.data import read_jsonl  # noqa: E402
from system_one.predictor import Predictor  # noqa: E402
from system_one.robustness import leak_test, order_report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", required=True, help="dynamic JSONL: context, question, options, target")
p.add_argument("--perms", type=int, default=6)
p.add_argument("--limit", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

pred = Predictor(a.checkpoint)
if not pred.model.dynamic:
    sys.exit("order_test needs a dynamic-mode checkpoint")
records = read_jsonl(a.data)[: a.limit]
rep = order_report(pred, records, a.perms, a.seed)
rep["leak_test"] = leak_test(pred, records)
rep["isolate_options"] = pred.cfg.decision.isolate_options
out = Path(a.checkpoint).parent / f"order_{Path(a.data).stem}"
out.mkdir(parents=True, exist_ok=True)
(out / "order_report.json").write_text(json.dumps(rep, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(6, 4))
for k, accs in rep["position_accuracy"].items():
    ax.plot(range(len(accs)), accs, "o-", label=f"{k} options")
ax.set_xlabel("position of the correct option")
ax.set_ylabel("accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(out / "accuracy_by_position.png", dpi=150)

print(f"questions {rep['questions']} x {rep['permutations']} orders | isolate_options={rep['isolate_options']}")
for k in ("accuracy", "flip_rate", "pairwise_flip_rate", "worst_position_accuracy", "position_gap",
          "confidence_std", "true_option_prob_std", "ece", "perm_average_accuracy"):
    print(f"{k:24s} {rep[k]:.4f}")
print("leak test (0 = isolated):", rep["leak_test"])
print(f"report written to {out}")
