import torch
import torch.nn as nn

from ..config import ModelConfig
from .layers import DecoderBlock, RMSNorm


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


class DecoderTransformer(nn.Module):
    """input_ids [B,N] -> hidden [B,N,d]"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.apply(_init)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)

    def layer_list(self) -> list:
        return list(self.blocks)

    def tail_modules(self) -> list:
        return [self.final_norm]


class PretrainedBackbone(nn.Module):
    """Hugging Face decoder without LM head (AutoModel). input_ids [B,N] -> hidden [B,N,d]"""

    def __init__(self, name: str, load_weights: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if load_weights:
            self.model = AutoModel.from_pretrained(name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(name))
        self.model.float()
        if not (hasattr(self.model, "layers") and hasattr(self.model, "norm")):
            raise ValueError(f"{name}: expected a Llama/Qwen-style decoder with .layers and .norm")
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.get_input_embeddings().num_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    def layer_list(self) -> list:
        return list(self.model.layers)

    def tail_modules(self) -> list:
        return [self.model.norm]
