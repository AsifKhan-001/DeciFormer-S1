import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.agent_router import ToolRouter

ENC = "sentence-transformers/all-MiniLM-L6-v2"


def read(paths):
    return [json.loads(l) for p in paths for l in open(p)]


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for name in ("calibrate", "eval", "route"):
    s = sub.add_parser(name)
    s.add_argument("--state", default="experiments/router_state.json")
    s.add_argument("--encoder", default=ENC)
    if name != "route":
        s.add_argument("--data", nargs="+", required=True)
    if name == "calibrate":
        s.add_argument("--target", type=float, default=0.05)
    if name == "route":
        s.add_argument("--task", required=True)
        s.add_argument("--tools", required=True, help="json file: {tool name: description}")
a = ap.parse_args()

r = ToolRouter(a.encoder)
if a.cmd == "calibrate":
    out = r.calibrate(read(a.data), a.target)
    r.save(a.state)
    print(json.dumps(out, indent=1))
    print("saved", a.state)
else:
    r.load(a.state)
    if a.cmd == "eval":
        print(json.dumps(r.evaluate(read(a.data)), indent=1))
    else:
        d = r.route(a.task, json.load(open(a.tools)))
        r.save(a.state)
        print(json.dumps({k: v for k, v in d.__dict__.items() if k != "scores"}, indent=1))
