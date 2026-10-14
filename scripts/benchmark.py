import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from system_one.predictor import Predictor  # noqa: E402
from system_one.utils import count_parameters, sync  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--iters", type=int, default=100)
p.add_argument("--warmup", type=int, default=10)
a = p.parse_args()

pred = Predictor(a.checkpoint)
dev = pred.device
text = "My login page is not opening in the WhatsApp application."
times = {"tokenize": [], "forward": [], "policy": [], "total": []}
for i in range(a.warmup + a.iters):
    t0 = perf_counter()
    batch, names = pred.encode(text)
    t1 = perf_counter()
    out = pred.forward(batch)
    sync(dev)
    t2 = perf_counter()
    pred.decide(out, names)
    t3 = perf_counter()
    if i >= a.warmup:
        for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
            times[k].append(v * 1000)

c = count_parameters(pred.model)
print(f"device: {dev} | params: {c['total']:,} | checkpoint: {Path(a.checkpoint).stat().st_size / 1e6:.1f} MB")
if dev.type == "mps":
    print(f"mps memory: {torch.mps.current_allocated_memory() / 1e6:.1f} MB")
if dev.type == "cuda":
    print(f"cuda memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
for k, v in times.items():
    print(f"{k:9s} p50 {np.percentile(v, 50):.2f} ms | p95 {np.percentile(v, 95):.2f} ms | p99 {np.percentile(v, 99):.2f} ms")
print(f"throughput: {1000 / np.mean(times['total']):.1f} queries/s (batch size 1)")
