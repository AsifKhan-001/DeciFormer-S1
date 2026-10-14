import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.config import load_config  # noqa: E402
from system_one.trainer import run_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--train-data", required=True)
p.add_argument("--val-data", required=True)
p.add_argument("--out-dir", default=None)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
cfg = load_config(a.config)
out = a.out_dir or f"experiments/{Path(a.config).stem}"
run_training(cfg, a.train_data, a.val_data, out, resume=a.resume)
print(f"saved to {out}")
