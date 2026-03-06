"""
Pretrain the base model with learnable RMSNorm scale γ instead of parameter-free norm.

Same CLI and behavior as scripts.base_train, but the model is GPTRMSNorm.
Run from the nanochat repo root (after copying gpt_rmsnorm.py into nanochat/):

  torchrun --standalone --nproc_per_node=8 -m scripts.base_train_rmsnorm -- \\
      --depth=12 --device-batch-size=32 --run=picochat_rmsnorm --model-tag=d12_rmsnorm
"""

import runpy
import os

# Patch GPT to GPTRMSNorm before base_train is loaded
import nanochat.gpt as gpt_module
from nanochat.gpt_rmsnorm import GPTRMSNorm

gpt_module.GPT = GPTRMSNorm

# Run the standard base_train script (same CLI, same loop; model is now GPTRMSNorm)
base_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_train.py")
runpy.run_path(base_train_path, run_name="__main__")
