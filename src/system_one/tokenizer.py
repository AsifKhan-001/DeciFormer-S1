import json
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"\[(?:DECISION|OPT_\d+)\]|\w+|[^\w\s]")


class SimpleTokenizer:
    kind = "simple"

    def __init__(self, vocab: list, max_options: int = 8):
        self.max_options = max_options
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["[PAD]"]
        self.unk_id = self.stoi["[UNK]"]
        self.decision_token_id = self.stoi["[DECISION]"]
        self.option_marker_ids = [self.stoi[f"[OPT_{i}]"] for i in range(max_options)]

    @staticmethod
    def specials(max_options: int) -> list:
        return ["[PAD]", "[UNK]", "[DECISION]"] + [f"[OPT_{i}]" for i in range(max_options)]

    @staticmethod
    def _split(text: str) -> list:
        return [t if t.startswith("[") else t.lower() for t in _TOKEN_RE.findall(text)]

    @classmethod
    def build(cls, texts, max_options: int = 8, min_freq: int = 1) -> "SimpleTokenizer":
        counts = Counter(t for x in texts for t in cls._split(x))
        sp = cls.specials(max_options)
        words = sorted(w for w, c in counts.items() if c >= min_freq and w not in sp)
        return cls(sp + words, max_options)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list:
        return [self.stoi.get(t, self.unk_id) for t in self._split(text)]

    def decode(self, ids) -> str:
        return " ".join(self.itos[i] for i in ids)

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text(
            json.dumps({"vocab": self.itos, "max_options": self.max_options}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SimpleTokenizer":
        d = json.loads((Path(path) / "tokenizer.json").read_text(encoding="utf-8"))
        return cls(d["vocab"], d["max_options"])


class HFTokenizer:
    kind = "hf"

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def encode(self, text: str) -> list:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def save(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.tok.save_pretrained(str(path))


def build_tokenizer(cfg, texts=None, load_dir=None):
    if cfg.data.tokenizer == "simple":
        if load_dir is not None:
            return SimpleTokenizer.load(load_dir)
        return SimpleTokenizer.build(texts, cfg.decision.max_options, cfg.data.min_freq)
    return HFTokenizer(str(load_dir) if load_dir is not None else cfg.model.pretrained_name)
