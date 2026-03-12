"""
Chat CLI that uses GPTSwiGLU instead of the baseline GPT.

Run e.g.:
  uv run python -m scripts.chat_cli_swiglu -i base -g d20_swiglu
"""

import runpy

import nanochat.gpt as gpt_module
from nanochat.gpt_swiglu import GPTSwiGLU
import nanochat.checkpoint_manager as cm

# Patch both the gpt module and checkpoint_manager to use GPTSwiGLU
gpt_module.GPT = GPTSwiGLU
cm.GPT = GPTSwiGLU

# Now run the regular chat_cli main
runpy.run_module("scripts.chat_cli", run_name="__main__")