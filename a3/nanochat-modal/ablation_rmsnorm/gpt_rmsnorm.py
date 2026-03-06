"""
Learnable RMSNorm ablation: GPT variant that adds learnable scale γ to RMSNorm.

Drop this file into the nanochat repo as nanochat/gpt_rmsnorm.py.
Train with: torchrun ... -m scripts.base_train_rmsnorm -- --depth=12 ...

Replaces parameter-free norm(x) = F.rms_norm(x, (c,)) with learnable γ:
  F.rms_norm(x, (c,), weight=γ) where γ = nn.Parameter(torch.ones(d_model))

Per LLaMA/Gemma/Mistral: allows rescaling individual channels post-normalization.
Adds 4 × n_embd params for the four main norm positions (post-embed, pre-attn, pre-mlp, post-block)
or (1 + 3*n_layer) × n_embd for full coverage. We use full coverage for consistency.
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

from nanochat.gpt import (
    GPTConfig,
    has_ve,
    apply_rotary_emb,
    CausalSelfAttention,
    GPT,
    Linear,
    MLP,
)


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale γ. Replaces parameter-free F.rms_norm(x, (c,))."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight.to(dtype=x.dtype), eps=self.eps)


class BlockRMSNorm(nn.Module):
    """Same as Block but uses RMSNorm (learnable γ) instead of parameter-free norm."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.pre_attn_norm = RMSNorm(config.n_embd)
        self.pre_mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(self.pre_attn_norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(self.pre_mlp_norm(x))
        return x


class GPTRMSNorm(GPT):
    """
    GPT with learnable RMSNorm scale γ instead of parameter-free norm.
    Same API as GPT; only the normalization has learnable parameters.
    """

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        # Replace Block with BlockRMSNorm
        self.transformer.h = nn.ModuleList([
            BlockRMSNorm(config, layer_idx) for layer_idx in range(config.n_layer)
        ])
        # Post-embed norm and post-block norms (used in forward loop)
        self.post_embed_norm = RMSNorm(config.n_embd)
        self.post_block_norms = nn.ModuleList([
            RMSNorm(config.n_embd) for _ in range(config.n_layer)
        ])

    @torch.no_grad()
    def init_weights(self):
        """Same as GPT.init_weights plus RMSNorm γ init to ones."""
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
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # RMSNorm γ: keep ones (neutral at init)
        for rn in [self.post_embed_norm] + list(self.post_block_norms):
            rn.weight.fill_(1.0)
        for block in self.transformer.h:
            block.pre_attn_norm.weight.fill_(1.0)
            block.pre_mlp_norm.weight.fill_(1.0)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def num_scaling_params(self):
        """Include RMSNorm params. Base would assert-fail due to our extra post_embed/post_block norms."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # post_embed_norm and post_block_norms are not in transformer.h
        rmsnorm = sum(p.numel() for p in self.post_embed_norm.parameters())
        rmsnorm += sum(p.numel() for p in self.post_block_norms.parameters())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + rmsnorm
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "rmsnorm": rmsnorm,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """Add RMSNorm params to AdamW group (like embeddings)."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect RMSNorm params
        rmsnorm_params = list(self.post_embed_norm.parameters()) + list(self.post_block_norms.parameters())
        for block in self.transformer.h:
            rmsnorm_params += list(block.pre_attn_norm.parameters())
            rmsnorm_params += list(block.pre_mlp_norm.parameters())

        # Matrix params: exclude RMSNorm (they're in transformer.h but we need to separate)
        matrix_params = []
        for block in self.transformer.h:
            matrix_params += list(block.attn.parameters())
            matrix_params += list(block.mlp.parameters())

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=rmsnorm_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = self.post_embed_norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            x = self.post_block_norms[i](x)

        softcap = 20
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits
