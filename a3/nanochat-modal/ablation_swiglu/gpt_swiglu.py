"""
SwiGLU ablation: GPT variant that uses SwiGLU activation in the MLP instead of ReLU².

Drop this file into the nanochat repo as nanochat/gpt_swiglu.py.
Train with: torchrun ... -m scripts.base_train_swiglu -- --depth=12 ...

SwiGLU: down(silu(gate(x)) * up(x))
- Hidden dim H = [8*n_embd/3 / 64] * 64 to match FLOPs with 4C two-matrix MLP (Shazeer 2020, LLaMA 2).
- 3 matrices: gate, up, down.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

from nanochat.gpt import (
    GPTConfig,
    norm,
    has_ve,
    apply_rotary_emb,
    CausalSelfAttention,
    GPT,
)


def _swiglu_hidden_size(n_embd: int) -> int:
    """SwiGLU intermediate size to match FLOPs with 4C two-matrix MLP. H = [8C/3 / 64] * 64."""
    return ((8 * n_embd // 3 + 63) // 64) * 64


class MLPSwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). 3 matrices for iso-FLOP comparison with 4C ReLU² MLP."""

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        self.hidden_size = _swiglu_hidden_size(n_embd)
        self.gate = nn.Linear(n_embd, self.hidden_size, bias=False)
        self.up = nn.Linear(n_embd, self.hidden_size, bias=False)
        self.down = nn.Linear(self.hidden_size, n_embd, bias=False)

    def forward(self, x):
        gate_x = self.gate(x)
        up_x = self.up(x)
        return self.down(F.silu(gate_x) * up_x)


class BlockSwiGLU(nn.Module):
    """Same as Block but uses MLPSwiGLU instead of MLP."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLPSwiGLU(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPTSwiGLU(GPT):
    """
    GPT that uses BlockSwiGLU (SwiGLU MLP) instead of Block (ReLU² MLP).
    Same API as GPT; only the MLP in each block is different.
    """

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        # Replace Block with BlockSwiGLU
        self.transformer.h = nn.ModuleList([
            BlockSwiGLU(config, layer_idx) for layer_idx in range(config.n_layer)
        ])

    @torch.no_grad()
    def init_weights(self):
        """Same as GPT.init_weights but init SwiGLU MLP (gate/up like c_fc, down like c_proj)."""
        # Embedding and unembedding (same as GPT)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.up.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.down.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
