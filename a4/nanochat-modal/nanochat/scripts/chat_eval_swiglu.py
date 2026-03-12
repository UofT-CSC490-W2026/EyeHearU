"""
Chat evaluation that uses GPTSwiGLU instead of the baseline GPT.

Monkey-patches GPT -> GPTSwiGLU, then runs the standard chat_eval main.
Same pattern as chat_cli_swiglu.py.

Run e.g.:
  torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval_swiglu -- -i sft --model-tag=d12_swiglu
"""

import runpy

import nanochat.gpt as gpt_module
from nanochat.gpt_swiglu import GPTSwiGLU
import nanochat.checkpoint_manager as cm

gpt_module.GPT = GPTSwiGLU
cm.GPT = GPTSwiGLU

runpy.run_module("scripts.chat_eval", run_name="__main__")
