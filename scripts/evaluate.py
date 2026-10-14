import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torch.utils.data import DataLoader  # noqa: E402

from system_one.data import DecisionDataset, make_collate  # noqa: E402
from system_one.losses import TotalLoss  # noqa: E402
from system_one.metrics import classification_report, save_plots  # noqa: E402
from system_one.policy import DecisionPolicy  # noqa: E402
from system_one.predictor import load_model  # noqa: E402
from system_one.trainer import collect  # noqa: E402
from system_one.utils import count_parameters  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--test-data", required=True)
p.add_argument("--out-dir", default=None)
a = p.parse_args()

model, tok, cfg, labels, device = load_model(a.checkpoint)
ds = DecisionDataset(a.test_data, tok, cfg)
loader = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=make_collate(tok.pad_id))
arr = collect(model, loader, device, TotalLoss(cfg.training.alpha_brier, cfg.training.beta_conf))
policy = DecisionPolicy(**asdict(cfg.policy))
rep = classification_report(arr, labels, policy, [r.get("difficulty") for r in ds.records])
rep["parameters"] = count_parameters(model)
rep["temperature"] = float(model.temperature.temperature.detach())
out = Path(a.out_dir or Path(a.checkpoint).parent / f"eval_{Path(a.test_data).stem}")
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(rep, indent=2))
save_plots(arr, rep, out)
keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "nll", "brier", "ece", "ece_softmax", "avg_confidence"]
print({k: round(rep[k], 4) for k in keys})
print("policy:", {k: (None if v is None else round(v, 4)) for k, v in rep["policy"].items()})
print("by difficulty:", rep.get("by_difficulty"))
print(f"report written to {out}")
