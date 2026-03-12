"""
SFT that uses GPTSwiGLU instead of the baseline GPT.

Monkey-patches GPT -> GPTSwiGLU, then runs the standard chat_sft main.
Same pattern as chat_cli_swiglu.py.

Run e.g.:
  torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft_swiglu -- --model-tag=d12_swiglu --model-step=2205
"""

import runpy

import nanochat.gpt as gpt_module
from nanochat.gpt_swiglu import GPTSwiGLU
import nanochat.checkpoint_manager as cm

gpt_module.GPT = GPTSwiGLU
cm.GPT = GPTSwiGLU

runpy.run_module("scripts.chat_sft", run_name="__main__")
