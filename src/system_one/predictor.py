from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from .config import Config
from .data import DEFAULT_QUESTION, encode_dynamic, encode_fixed, make_collate
from .model.system_one import SystemOneModel
from .policy import DecisionPolicy
from .tokenizer import build_tokenizer
from .trainer import load_checkpoint, model_inputs, to_device
from .utils import get_device


def load_model(checkpoint, device="auto"):
    ckpt = load_checkpoint(checkpoint)
    cfg = Config.from_dict(ckpt["config"])
    tok = build_tokenizer(cfg, load_dir=Path(checkpoint).parent / "tokenizer")
    cfg.model.vocab_size = tok.vocab_size
    dev = get_device(device)
    model = SystemOneModel(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return model, tok, cfg, {int(k): v for k, v in ckpt["labels"].items()}, dev


class Predictor:
    def __init__(self, checkpoint, device: str = "auto"):
        self.model, self.tok, self.cfg, self.labels, self.device = load_model(checkpoint, device)
        self.policy = DecisionPolicy(**asdict(self.cfg.policy))
        self.collate = make_collate(self.tok.pad_id)
        self.warmup()

    def _encode_one(self, text, options=None, question=DEFAULT_QUESTION):
        ml = self.cfg.model.max_seq_len
        if self.model.dynamic:
            options = options or [self.labels[i].replace("_", " ") for i in range(len(self.labels))]
            enc = encode_dynamic(self.tok, text, question, options, ml, self.cfg.decision.max_options)
            return enc, list(options)
        return encode_fixed(self.tok, text, ml), [self.labels[i] for i in range(self.cfg.model.num_classes)]

    def encode(self, text: str, options: list = None, question: str = DEFAULT_QUESTION):
        enc, names = self._encode_one(text, options, question)
        return to_device(self.collate([{**enc, "target": 0}]), self.device), names

    @torch.inference_mode()
    def forward(self, batch):
        return self.model(**model_inputs(batch))

    def decide(self, out, names: list, i: int = 0) -> dict:
        probs = out.probabilities[i, : len(names)].tolist()
        conf = float(out.confidence[i, 0]) if out.confidence is not None else max(probs)
        entropy, margin = float(out.entropy[i, 0]), float(out.top2_margin[i, 0])
        pid = int(out.prediction[i])
        return {
            "prediction": names[pid], "prediction_id": pid,
            "scores": dict(zip(names, probs)),
            "confidence": conf, "entropy": entropy, "top2_margin": margin,
            "action": self.policy(conf, margin, entropy)[0],
        }

    def predict(self, text: str, options: list = None, question: str = DEFAULT_QUESTION) -> dict:
        t0 = perf_counter()
        batch, names = self.encode(text, options, question)
        result = self.decide(self.forward(batch), names)
        result["latency_ms"] = (perf_counter() - t0) * 1000
        return result

    def predict_batch(self, requests: list, batch_size: int = 64) -> list:
        """requests: [{"text"|"context": str, "question": str, "options": [str]}] -> results in the same order."""
        enc = []
        for r in requests:
            e, names = self._encode_one(r.get("text", r.get("context", "")), r.get("options"),
                                        r.get("question", DEFAULT_QUESTION))
            enc.append((e, names))
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]["input_ids"]))
        results = [None] * len(enc)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            batch = to_device(self.collate([{**enc[i][0], "target": 0} for i in idx]), self.device)
            out = self.forward(batch)
            for row, i in enumerate(idx):
                results[i] = self.decide(out, enc[i][1], row)
        return results

    def warmup(self, n: int = 3) -> None:
        for _ in range(n):
            self.predict("warm up", None if not self.model.dynamic else ["a", "b"], "Which option?")
