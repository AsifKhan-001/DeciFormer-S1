import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..metrics import calculate_entropy, calculate_top2_margin
from .backbone import DecoderTransformer, PretrainedBackbone
from .heads import CalibrationHead, DecisionHead, DecisionPooling, OptionScorer, TemperatureScaler
from .lora import apply_lora


@dataclass
class SystemOneOutput:
    logits: Tensor
    scaled_logits: Tensor
    probabilities: Tensor
    prediction: Tensor
    entropy: Tensor
    top2_margin: Tensor
    representation: Tensor
    confidence: Optional[Tensor] = None


class SystemOneModel(nn.Module):
    """input_ids [B,N] -> backbone [B,N,d] -> decision head [B,K] -> temperature -> softmax -> confidence."""

    def __init__(self, cfg: Config, load_pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        m, dc = cfg.model, cfg.decision
        if m.backbone == "scratch":
            if m.vocab_size <= 0:
                raise ValueError("model.vocab_size must be set (tokenizer vocab) for the scratch backbone")
            self.backbone = DecoderTransformer(m)
        else:
            self.backbone = PretrainedBackbone(m.pretrained_name, load_weights=load_pretrained)
            if m.vocab_size > self.backbone.vocab_size:
                raise ValueError(f"tokenizer vocab {m.vocab_size} > embedding rows {self.backbone.vocab_size}")
        if cfg.training.lora_rank > 0:
            apply_lora(self.backbone, cfg.training.lora_rank, cfg.training.lora_alpha)
        d = self.backbone.hidden_size
        self.dynamic = dc.mode == "dynamic"
        self.pooling = DecisionPooling(dc.pooling, d)
        self.decision_head = None if self.dynamic else DecisionHead(d, m.num_classes, m.dropout)
        self.option_scorer = OptionScorer(d, m.dropout) if self.dynamic else None
        self.temperature = TemperatureScaler(dc.temperature_init, dc.min_temperature, dc.learn_temperature)
        self.calibration_head = (
            CalibrationHead(d, cfg.calibration.hidden_size, m.dropout) if cfg.calibration.enabled else None)
        self.backbone_frozen = False

    @property
    def num_outputs(self) -> int:
        return self.cfg.decision.max_options if self.dynamic else self.cfg.model.num_classes

    def set_backbone_trainable(self, fraction: float) -> None:
        bb = self.backbone
        for n, p in bb.named_parameters():
            p.requires_grad = "lora_" in n
        if fraction >= 1.0:
            for p in bb.parameters():
                p.requires_grad = True
        elif fraction > 0:
            layers = bb.layer_list()
            k = max(1, math.ceil(len(layers) * fraction))
            for mod in layers[-k:] + bb.tail_modules():
                for p in mod.parameters():
                    p.requires_grad = True
        self.backbone_frozen = not any(p.requires_grad for p in bb.parameters())

    def forward(self, input_ids: Tensor, attention_mask: Tensor, option_indices: Tensor = None,
                option_mask: Tensor = None, decision_position: Tensor = None) -> SystemOneOutput:
        if self.backbone_frozen:
            with torch.no_grad():
                h = self.backbone(input_ids, attention_mask)
        else:
            h = self.backbone(input_ids, attention_mask)
        h = h.float()

        if not self.dynamic:
            rep = self.pooling(h, attention_mask, decision_position)
            logits = self.decision_head(rep)
            num_options = logits.size(-1)
        else:
            gathered = h.gather(1, option_indices.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            logits = self.option_scorer(gathered).masked_fill(~option_mask, -1e9)
            num_options = option_mask.sum(1)

        scaled = self.temperature(logits)
        probs = scaled.softmax(-1)
        if self.dynamic:
            rep = (probs.unsqueeze(-1) * gathered).sum(1)
        entropy = calculate_entropy(probs, num_options)
        margin = calculate_top2_margin(probs)
        confidence = None
        if self.calibration_head is not None:
            confidence = self.calibration_head(rep.detach(), entropy.detach())
        return SystemOneOutput(logits, scaled, probs, probs.argmax(-1), entropy, margin, rep, confidence)
