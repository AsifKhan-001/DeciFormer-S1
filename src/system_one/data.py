import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

DEFAULT_QUESTION = "Which team should handle this request?"


def read_jsonl(path) -> list:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found. Run from the project root and generate the data first (scripts/make_data.py or scripts/make_multitask.py).")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_texts(rec: dict) -> list:
    if "options" in rec:
        return [rec.get("context", ""), rec.get("question", "")] + list(rec["options"])
    return [rec["query"]]


def encode_fixed(tok, query: str, max_len: int) -> dict:
    pre, suf = tok.encode("Query:"), tok.encode("\n[DECISION]")
    q = tok.encode(query)[: max(1, max_len - len(pre) - len(suf))]
    ids = pre + q + suf
    return {"input_ids": ids, "decision_position": len(ids) - 1}


def encode_dynamic(tok, context: str, question: str, options: list, max_len: int, max_options: int,
                   shared_marker: bool = False) -> dict:
    """shared_marker: every option ends with the same [OPT_0] token, so the slot index is not visible to the model.
    segment_ids: -1 = shared prompt, k = tokens of option k. position_ids restart at the prompt end for every option."""
    if not 2 <= len(options) <= max_options:
        raise ValueError(f"need 2..{max_options} options, got {len(options)}")
    opt_ids, local, lens = [], [], []
    for k, o in enumerate(options):
        piece = tok.encode(f"Candidate: {o}") + tok.encode("[OPT_0]" if shared_marker else f"[OPT_{k}]")
        opt_ids += piece
        lens.append(len(piece))
        local.append(len(opt_ids) - 1)
    qs = tok.encode(f"\nQuestion: {question}\n")
    room = max_len - len(opt_ids) - len(qs)
    if room < 4:
        raise ValueError("options too long for max_seq_len")
    head = tok.encode(f"Context: {context}")[:room] + qs
    hl = len(head)
    seg, pos = [-1] * hl, list(range(hl))
    for k, n in enumerate(lens):
        seg += [k] * n
        pos += list(range(hl, hl + n))
    ids = head + opt_ids
    return {"input_ids": ids, "option_positions": [hl + p for p in local], "decision_position": len(ids) - 1,
            "segment_ids": seg, "position_ids": pos}


class DecisionDataset(Dataset):
    """pair=True (dynamic mode only): every item also carries a randomly shuffled copy of its options ("alt")."""

    def __init__(self, path, tok, cfg, pair: bool = False):
        self.records = read_jsonl(path)
        self.tok, self.cfg, self.pair = tok, cfg, pair
        self.rng = random.Random(cfg.training.seed)
        ml, mo = cfg.model.max_seq_len, cfg.decision.max_options
        self.items = []
        for r in self.records:
            if cfg.decision.mode == "dynamic":
                e = encode_dynamic(tok, r.get("context", ""), r.get("question", DEFAULT_QUESTION),
                                   r["options"], ml, mo, cfg.decision.isolate_options)
                e["target"] = int(r["target"])
            else:
                e = encode_fixed(tok, r["query"], ml)
                e["target"] = int(r["label"])
                if not 0 <= e["target"] < cfg.model.num_classes:
                    raise ValueError(f"label {e['target']} outside [0,{cfg.model.num_classes})")
            self.items.append(e)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        if not self.pair:
            return item
        r = self.records[i]
        perm = list(range(len(r["options"])))
        self.rng.shuffle(perm)
        alt = encode_dynamic(self.tok, r.get("context", ""), r.get("question", DEFAULT_QUESTION),
                             [r["options"][p] for p in perm], self.cfg.model.max_seq_len, self.cfg.decision.max_options,
                             self.cfg.decision.isolate_options)
        alt["target"] = perm.index(int(r["target"]))
        alt["perm"] = perm
        return {**item, "alt": alt}


def make_collate(pad_id: int):
    def build(items: list) -> dict:
        n = max(len(b["input_ids"]) for b in items)
        ids = torch.full((len(items), n), pad_id, dtype=torch.long)
        mask = torch.zeros(len(items), n, dtype=torch.long)
        seg = torch.full((len(items), n), -3, dtype=torch.long)
        pos = torch.zeros(len(items), n, dtype=torch.long)
        for i, b in enumerate(items):
            length = len(b["input_ids"])
            ids[i, :length] = torch.tensor(b["input_ids"])
            mask[i, :length] = 1
            if "segment_ids" in b:
                seg[i, :length] = torch.tensor(b["segment_ids"])
                pos[i, :length] = torch.tensor(b["position_ids"])
        out = {"input_ids": ids, "attention_mask": mask,
               "target": torch.tensor([b["target"] for b in items]),
               "decision_position": torch.tensor([b["decision_position"] for b in items])}
        if "segment_ids" in items[0]:
            out["segment_ids"], out["position_ids"] = seg, pos
        if "option_positions" in items[0]:
            k = max(len(b["option_positions"]) for b in items)
            op = torch.zeros(len(items), k, dtype=torch.long)
            om = torch.zeros(len(items), k, dtype=torch.bool)
            for i, b in enumerate(items):
                kk = len(b["option_positions"])
                op[i, :kk] = torch.tensor(b["option_positions"])
                om[i, :kk] = True
            out["option_indices"], out["option_mask"] = op, om
        return out

    def collate(batch: list) -> dict:
        if "alt" not in batch[0]:
            return build(batch)
        items = list(batch) + [b["alt"] for b in batch]
        out = build(items)
        perm = torch.arange(out["option_indices"].size(1)).repeat(len(items), 1)
        for j, b in enumerate(batch):
            p = b["alt"]["perm"]
            perm[len(batch) + j, : len(p)] = torch.tensor(p)
        out["perm"] = perm
        return out

    return collate
