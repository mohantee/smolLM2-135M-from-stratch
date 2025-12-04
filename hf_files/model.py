# model.py
# SmolLM2 / LLaMA-like model implementation
# - GQA (grouped query attention)
# - RoPE with precomputed cos/sin and start_pos support (cache-friendly)
# - RMSNorm, Gated MLP (SiLU)
# - KV cache support for autoregressive inference
#
# Notes:
# - Default config values match SmolLM2-135M (hidden_size=576, n_layer=30, n_head=9, n_kv_heads=3, intermediate_size=1536)
# - Forward semantics:
#     logits = model(input_ids)                      # training/simple usage
#     logits, past = model(input_ids, use_cache=True) # inference, returns past_key_values
#     model(input_ids, past_key_values=past, use_cache=True)  # continue generation
#
# Tested shapes carefully; follow GQA shapes exactly.

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmolConfig:
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    intermediate_size: int = 1536
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    rope_theta: float = 10000.0
    # additional options
    use_cache: bool = True


# -------------------------
# Small building blocks
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x (..., dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


def get_activation(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


# -------------------------
# RoPE (precomputed cos/sin)
# -------------------------
class RoPE(nn.Module):
    """
    Rotary positional embeddings implemented with precomputed cos/sin buffers.
    This is numerically stable and cache-friendly. It supports 'start_pos' so that
    during autoregressive decoding you can provide the offset for the current chunk.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0, device=None, dtype=torch.float32):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2)
        # precompute (max_seq_len, head_dim/2)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        # produce cos and sin of shape (max_seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        cos = emb.cos()  # (max_seq_len, head_dim)
        sin = emb.sin()
        # Register as buffers so they move with model.device
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def rotate_half(x):
        # x: (..., head_dim)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x: torch.Tensor, start_pos: int = 0):
        """
        x: (batch, n_heads, seq_len, head_dim)
        returns: x with RoPE applied
        """
        seq_len = x.shape[2]
        cos = self.cos[start_pos:start_pos + seq_len].to(x.device).unsqueeze(0).unsqueeze(0)  # (1,1,seq,head_dim)
        sin = self.sin[start_pos:start_pos + seq_len].to(x.device).unsqueeze(0).unsqueeze(0)
        # apply: x * cos + rotate_half(x) * sin
        return x * cos + self.rotate_half(x) * sin


# -------------------------
# Attention with GQA (Grouped Query Attention)
# -------------------------
class GroupedQueryAttention(nn.Module):
    """
    GQA implementation:
      - q_proj outputs (B, T, n_heads * head_dim)
      - k_proj/v_proj output (B, T, n_kv_heads * head_dim)
      - we reshape and expand k/v to match q heads by group repetition (logical grouping)
      - supports past_key_values for efficient incremental decoding (stores k/v in kv-head shape)
    """
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.d_model = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        assert self.d_model % self.n_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.head_dim = self.d_model // self.n_heads
        assert self.n_heads % self.n_kv == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.group_size = self.n_heads // self.n_kv  # how many q-heads per kv-head

        # projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE placeholder (set externally by block with proper head_dim and max_seq_len)
        self.rope: Optional[RoPE] = None

    def _shape_q(self, q):
        # q: (B, T, d_model) -> (B, n_heads, T, head_dim)
        B, T, _ = q.size()
        q = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        return q

    def _shape_kv(self, kv, is_k=True):
        # kv: (B, T, n_kv*head_dim) -> (B, n_kv, T, head_dim)
        B, T, _ = kv.size()
        kv = kv.view(B, T, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        return kv  # (B, n_kv, T, head_dim)

    def _expand_kv_to_heads(self, kv):
        # kv: (B, n_kv, T, head_dim) -> (B, n_heads, T, head_dim) by repeating groups
        if self.group_size == 1:
            return kv.repeat_interleave(1, dim=1)  # trivial
        # Efficient reshape: expand along head axis
        B, n_kv, T, hd = kv.shape
        # repeat each kv head group_size times along head axis
        kv = kv.unsqueeze(2)  # (B, n_kv, 1, T, hd)
        kv = kv.repeat(1, 1, self.group_size, 1, 1)  # (B, n_kv, group_size, T, hd)
        kv = kv.view(B, n_kv * self.group_size, T, hd)  # (B, n_heads, T, hd)
        return kv

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                start_pos: int = 0):
        """
        x: (B, T, d_model)
        past_key_value: tuple(k_cache, v_cache)
            k_cache: (B, n_kv, cached_len, head_dim)
            v_cache: (B, n_kv, cached_len, head_dim)
        use_cache: whether to return updated caches
        start_pos: offset to apply when using RoPE (for cached generation)
        returns:
            out: (B, T, d_model)
            (optional) present_kv: (k_cat, v_cat) each in kv-head shape (B, n_kv, total_len, head_dim)
        """
        B, T, _ = x.shape

        # projections
        q = self.q_proj(x)  # (B, T, n_heads*hd)
        k = self.k_proj(x)  # (B, T, n_kv*hd)
        v = self.v_proj(x)  # (B, T, n_kv*hd)

        # reshape to head dims
        q = self._shape_q(q)  # (B, n_heads, T, hd)
        k = self._shape_kv(k)  # (B, n_kv, T, hd)
        v = self._shape_kv(v)  # (B, n_kv, T, hd)

        # apply RoPE to q and k
        if self.rope is not None:
            # RoPE expects (B, n_heads, seq, head_dim) for q
            q = self.rope.apply_rotary(q, start_pos=start_pos)  # (B, n_heads, T, hd)
            # For k, it is in (B, n_kv, T, hd) - to reuse same rotor we need to expand frequencies
            # We'll expand k to n_heads, apply rope, then compress back to n_kv for caching/efficiency only if needed.
            # Simpler and correct approach: expand k to n_heads, apply rope, then fold back by taking every group_size-th head.
            k_expanded = self._expand_kv_to_heads(k)  # (B, n_heads, T, hd)
            k_expanded = self.rope.apply_rotary(k_expanded, start_pos=start_pos)
            # collapse back to (B, n_kv, T, hd) by averaging groups (but to be exact we should pick the first of each group;
            # however since expansion is repetition, collapsing by reshaping recovers original k with per-head rotations applied identically)
            # reshape back
            k = k_expanded.view(B, self.n_kv, self.group_size, T, self.head_dim)[:, :, 0, :, :]  # take first in group
            # NOTE: above trick keeps identical rotations across repeated kv heads
        # else: no RoPE applied (should not happen if rope is set)

        # If past kv present (for generation), append along time dimension in kv-head space
        if past_key_value is not None:
            past_k, past_v = past_key_value  # (B, n_kv, prev_len, hd)
            # concatenate on time dim
            k = torch.cat([past_k, k], dim=2)  # (B, n_kv, total_len, hd)
            v = torch.cat([past_v, v], dim=2)

        # Now expand kv to full heads (for attention computation)
        k_exp = self._expand_kv_to_heads(k)  # (B, n_heads, total_len, hd)
        v_exp = self._expand_kv_to_heads(v)  # (B, n_heads, total_len, hd)

        # q: (B, n_heads, T, hd)
        # k_exp: (B, n_heads, S, hd)
        # compute attention
        q_scaled = q / math.sqrt(self.head_dim)

        # (B, n_heads, T, S)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", q_scaled, k_exp)

        # causal mask (allow only attending to <= current position in combined kv sequence)
        # S = k_exp.shape[2]
        S = k_exp.shape[2]
        device = attn_scores.device
        # build causal mask where query positions (0..T-1) attend to keys (S-T..S-1) appropriately.
        # We assume k_exp contains all past + current keys in order. For training (no past) S == T and mask is lower-triangular.
        # We'll construct a mask that allows each query index qi to attend to keys indices <= (start_pos + qi) (if start_pos used).
        # Simpler robust approach: create a causal mask of shape (T, S) where for q at idx i, allowed keys are <= (S - T + i)
        # offset = S - T  (past length)
        offset = S - T
        idxs_q = torch.arange(T, device=device)[:, None]  # (T,1)
        idxs_k = torch.arange(S, device=device)[None, :]  # (1,S)
        # allow where idxs_k <= offset + idxs_q
        causal = idxs_k <= (offset + idxs_q)
        attn_scores = attn_scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, n_heads, T, S)

        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_probs, v_exp)  # (B,n_heads,T,hd)

        # merge heads: (B, T, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        out = self.o_proj(attn_output)

        present = None
        if use_cache:
            # return k and v in kv-head shape for future caching (B, n_kv, total_len, hd)
            present = (k, v)  # already in kv-head shape

        if use_cache:
            return out, present
        return out


# -------------------------
# FeedForward (Gated)
# -------------------------
class GatedMLP(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.d_model = cfg.hidden_size
        self.hidden = cfg.intermediate_size
        # gate, up, down
        self.gate_proj = nn.Linear(self.d_model, self.hidden, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.hidden, bias=False)
        self.down_proj = nn.Linear(self.hidden, self.d_model, bias=False)
        self.act = get_activation(cfg.hidden_act)

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# -------------------------
# Transformer Block
# -------------------------
class SmolBlock(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = GatedMLP(cfg)

    def forward(self, x: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                start_pos: int = 0):
        # x: (B, T, d_model)
        # attn_norm
        normed = self.attn_norm(x)
        # ensure RoPE instance exists and head_dim matches
        if getattr(self.attn, "rope", None) is None:
            # create and attach RoPE with head_dim and a sensible max length
            head_dim = self.attn.head_dim
            max_seq = 8192
            self.attn.rope = RoPE(head_dim=head_dim, max_seq_len=max_seq, theta=10000.0)
        # attention
        if use_cache:
            attn_out, present = self.attn(normed, past_key_value=past_key_value, use_cache=True, start_pos=start_pos)
        else:
            attn_out = self.attn(normed, past_key_value=past_key_value, use_cache=False, start_pos=start_pos)
            present = None

        x = x + attn_out
        # ffn
        x = x + self.mlp(self.ffn_norm(x))

        if use_cache:
            return x, present
        return x


# -------------------------
# Full Model
# -------------------------
class SmolLM(nn.Module):
    def __init__(self, cfg: SmolConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([SmolBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # weight tying
        try:
            self.lm_head.weight = self.embed_tokens.weight
        except Exception:
            pass

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # LLaMA/SmolLM initialization
        if isinstance(module, nn.Linear):
            # Special rule for attention output projection
            if hasattr(self, "layers") and any(module is layer.attn.o_proj for layer in self.layers):
                # LLaMA/SmolLM2 style: smaller std for o_proj
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(self.cfg.num_hidden_layers * self.cfg.hidden_size)
                )
            else:
                # All other linear layers
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(self.cfg.hidden_size)
                )

        elif isinstance(module, nn.Embedding):
            # Standard embedding init
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

        elif isinstance(module, RMSNorm):
            # RMSNorm always starts with all-ones weight
            nn.init.ones_(module.weight)


    def forward(self,
                input_ids: torch.LongTensor,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False):
        """
        input_ids: (B, T)
        past_key_values: list length = n_layers of (k_cache, v_cache) each in kv-head shape (B, n_kv, cached_len, hd)
        use_cache: if True returns (logits, new_past_key_values)
        """
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)  # (B, T, d_model)

        present_kvs = [] if use_cache else None
        start_pos = 0
        # compute start_pos (sum of cached lengths for first layer if provided)
        if past_key_values is not None and len(past_key_values) > 0:
            # each element is (k_cache, v_cache) => k_cache.shape = (B, n_kv, cached_len, hd)
            # get cached_len from first layer
            cached_len = past_key_values[0][0].shape[2]
            start_pos = cached_len

        for i, layer in enumerate(self.layers):
            pkv = None
            if past_key_values is not None:
                pkv = past_key_values[i]
            if use_cache:
                x, present = layer(x, past_key_value=pkv, use_cache=True, start_pos=start_pos)
                present_kvs.append(present)
            else:
                x = layer(x, past_key_value=pkv, use_cache=False, start_pos=start_pos)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        if use_cache:
            return logits, present_kvs
        return logits


