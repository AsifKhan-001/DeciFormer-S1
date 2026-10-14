import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class RMSNorm(nn.Module):
    """[..., d] -> [..., d]"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotates q,k [B,H,N,D]; v is never rotated."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        n = q.size(-2)
        cos, sin = self.cos[:n].to(q.dtype), self.sin[:n].to(q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """x [B,N,d], attention_mask [B,N] (1 = real token) -> [B,N,d]"""

    def __init__(self, d: int, heads: int, max_seq_len: int, theta: float, dropout: float):
        super().__init__()
        assert d % heads == 0, "hidden_size % num_heads != 0"
        self.h, self.dh, self.dropout = heads, d // heads, dropout
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.rope = RotaryEmbedding(self.dh, max_seq_len, theta)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        b, n, _ = x.shape
        split = lambda t: t.view(b, n, self.h, self.dh).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        q, k = self.rope(q, k)
        allow = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None]
        if attention_mask is not None:
            allow = allow & attention_mask.bool()[:, None, None, :]
            allow = allow | torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allow, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class SwiGLU(nn.Module):
    """[B,N,d] -> [B,N,d]"""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_size
        self.attn_norm = RMSNorm(d)
        self.attention = CausalSelfAttention(d, cfg.num_heads, cfg.max_seq_len, cfg.rope_theta, cfg.dropout)
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))
