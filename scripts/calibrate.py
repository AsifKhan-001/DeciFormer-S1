import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torch.utils.data import DataLoader  # noqa: E402

from system_one.calibration import fit_temperature, refit_calibration_head  # noqa: E402
from system_one.data import DecisionDataset, make_collate  # noqa: E402
from system_one.metrics import quick_metrics  # noqa: E402
from system_one.predictor import load_model  # noqa: E402
from system_one.trainer import collect, load_checkpoint, save_checkpoint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--val-data", required=True)
a = p.parse_args()

model, tok, cfg, labels, device = load_model(a.checkpoint)
ds = DecisionDataset(a.val_data, tok, cfg)
loader = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=make_collate(tok.pad_id))
before = quick_metrics(collect(model, loader, device))
arr = collect(model, loader, device)
t = fit_temperature(arr["raw_logits"], arr["target"], cfg.decision.min_temperature)
model.temperature.set_temperature(t)
if model.calibration_head is not None:
    refit_calibration_head(model, loader, device)
after = quick_metrics(collect(model, loader, device))
print(f"temperature -> {t:.4f}")
for k in ("val_nll", "val_ece", "val_brier"):
    print(f"{k}: {before[k]:.4f} -> {after[k]:.4f}")
out = Path(a.checkpoint).parent / "best_calibrated.pt"
save_checkpoint(out, model, cfg, labels, load_checkpoint(a.checkpoint)["state"])
print(f"saved {out}")
