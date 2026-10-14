import json
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


def encode_dynamic(tok, context: str, question: str, options: list, max_len: int, max_options: int) -> dict:
    if not 2 <= len(options) <= max_options:
        raise ValueError(f"need 2..{max_options} options, got {len(options)}")
    opt_ids, local = [], []
    for k, o in enumerate(options):
        opt_ids += tok.encode(f"Candidate: {o}") + tok.encode(f"[OPT_{k}]")
        local.append(len(opt_ids) - 1)
    qs = tok.encode(f"\nQuestion: {question}\n")
    room = max_len - len(opt_ids) - len(qs)
    if room < 4:
        raise ValueError("options too long for max_seq_len")
    head = tok.encode(f"Context: {context}")[:room] + qs
    ids = head + opt_ids
    return {"input_ids": ids, "option_positions": [len(head) + p for p in local], "decision_position": len(ids) - 1}


class DecisionDataset(Dataset):
    def __init__(self, path, tok, cfg):
        self.records = read_jsonl(path)
        ml, mo = cfg.model.max_seq_len, cfg.decision.max_options
        self.items = []
        for r in self.records:
            if cfg.decision.mode == "dynamic":
                e = encode_dynamic(tok, r.get("context", ""), r.get("question", DEFAULT_QUESTION),
                                   r["options"], ml, mo)
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
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch: list) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros(len(batch), n, dtype=torch.long)
        for i, b in enumerate(batch):
            length = len(b["input_ids"])
            ids[i, :length] = torch.tensor(b["input_ids"])
            mask[i, :length] = 1
        out = {"input_ids": ids, "attention_mask": mask,
               "target": torch.tensor([b["target"] for b in batch]),
               "decision_position": torch.tensor([b["decision_position"] for b in batch])}
        if "option_positions" in batch[0]:
            k = max(len(b["option_positions"]) for b in batch)
            pos = torch.zeros(len(batch), k, dtype=torch.long)
            om = torch.zeros(len(batch), k, dtype=torch.bool)
            for i, b in enumerate(batch):
                kk = len(b["option_positions"])
                pos[i, :kk] = torch.tensor(b["option_positions"])
                om[i, :kk] = True
            out["option_indices"], out["option_mask"] = pos, om
        return out

    return collate
