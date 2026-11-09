import json, os, random, re

SRC = "data/raw/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"
DES = "data/raw/MetaTool/dataset/plugin_des.json"
OUT = "data/metatool"
random.seed(0)

des = json.load(open(DES))
pat = re.compile(r"tool name:\s*(.+?),\s*tool description:")


def label(name):
    d = des.get(name, "")
    d = d if isinstance(d, str) else " ".join(d)
    return f"{name}: {' '.join(d.split()[:12])}"


rows, bad = [], {"no_names": 0, "gold_missing": 0}
for r in json.load(open(SRC)):
    names = [n.strip().strip("\"'") for n in pat.findall(r["action_prompt"])]
    names = list(dict.fromkeys(names))
    gold = r["tool"].strip()
    if not names:
        bad["no_names"] += 1
        continue
    if gold not in names:
        bad["gold_missing"] += 1
        continue
    others = [n for n in names if n != gold]
    names = [gold] + random.sample(others, min(len(others), random.randint(1, 5)))
    random.shuffle(names)
    rows.append({"context": r["query"], "question": "Which tool best handles this request?",
                 "options": [label(n) for n in names], "target": names.index(gold), "tool": gold})
print("skipped", bad)

tools = sorted({r["tool"] for r in rows})
random.shuffle(tools)
n = len(tools)
split = {t: "train" for t in tools[: int(n * .7)]}
split.update({t: "val" for t in tools[int(n * .7): int(n * .8)]})
split.update({t: "test" for t in tools[int(n * .8):]})

os.makedirs(OUT, exist_ok=True)
for s in ("train", "val", "test"):
    part = [{k: v for k, v in r.items() if k != "tool"} for r in rows if split[r["tool"]] == s]
    with open(f"{OUT}/{s}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in part)
    print(s, len(part))
print("tools", n, "kept queries", len(rows))