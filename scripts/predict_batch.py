import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.predictor import Predictor  # noqa: E402
from system_one.utils import sync  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--input", required=True, help="JSONL: {context|text, question, options}")
p.add_argument("--output", default=None)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

reqs = [json.loads(line) for line in open(a.input, encoding="utf-8") if line.strip()]
if a.limit:
    reqs = reqs[: a.limit]
pred = Predictor(a.checkpoint)
t0 = perf_counter()
res = pred.predict_batch(reqs, a.batch_size)
sync(pred.device)
dt = perf_counter() - t0
if a.output:
    with open(a.output, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
print(f"{len(reqs)} queries on {pred.device} in {dt * 1000:.1f} ms | {dt * 1000 / len(reqs):.2f} ms/query | {len(reqs) / dt:.0f} queries/s")
if "target" in reqs[0]:
    ok = sum(r["prediction_id"] == q["target"] for r, q in zip(res, reqs))
    print(f"accuracy on input: {ok / len(reqs):.4f}")
