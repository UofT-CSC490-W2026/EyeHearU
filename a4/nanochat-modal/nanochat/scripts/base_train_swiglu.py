"""
Pretrain the base model with SwiGLU MLP instead of ReLU².

Same CLI and behavior as scripts.base_train, but the model is GPTSwiGLU.
Run from the nanochat repo root (after copying gpt_swiglu.py into nanochat/):

  torchrun --standalone --nproc_per_node=8 -m scripts.base_train_swiglu -- \\
      --depth=12 --device-batch-size=32 --run=picochat_swiglu --model-tag=d12_swiglu
"""

import runpy
import os

# Patch GPT to GPTSwiGLU before base_train is loaded, so "from nanochat.gpt import GPT" yields GPTSwiGLU
import nanochat.gpt as gpt_module
from nanochat.gpt_swiglu import GPTSwiGLU

gpt_module.GPT = GPTSwiGLU

# Run the standard base_train script (same CLI, same loop; model is now SwiGLU)
base_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_train.py")
runpy.run_path(base_train_path, run_name="__main__")
