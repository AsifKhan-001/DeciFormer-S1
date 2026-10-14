import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.predictor import Predictor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--text", required=True)
p.add_argument("--question", default="Which option is correct?")
p.add_argument("--options", default=None, help="comma-separated candidates (dynamic mode)")
a = p.parse_args()
opts = [o.strip() for o in a.options.split(",")] if a.options else None
print(json.dumps(Predictor(a.checkpoint).predict(a.text, opts, a.question), indent=2))