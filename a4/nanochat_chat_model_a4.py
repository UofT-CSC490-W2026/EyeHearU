# =============================================================================
# A4 CONFIGURATION (same as a3 for fair comparison)
# =============================================================================

DEPTH = 12
# Same as a3: pretrain used GPU_PRETRAIN="H100:8", SFT/RL uses GPU_FINETUNE="H100:4"
GPU_FINETUNE = "H100:4"
FINETUNE_TIMEOUT_SEC = 60 * 60 * 4    # 4 hours (combined-reward RL needs more time)
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"
BASE_DIR = "/data/.cache/nanochat"

# Pretrained checkpoint from a3: use baseline GPT model for direct comparability
# to Karpathy's original nanochat pipeline (no monkey-patching needed)
A4_MODEL_TAG = "d12"
A4_MODEL_STEP = "2205"   # pretrain step from a3 d12 baseline

# W&B run names (set to "dummy" to disable W&B):
#   Task 1 = original config SFT + RL
#   Task 2 = SFT + RL with additional datasets (e.g. OpenHermes)
WANDB_RUN_TASK1_SFT = "a4_task1_sft"
WANDB_RUN_TASK1_RL  = "a4_task1_rl"
WANDB_RUN_TASK2_SFT = "a4_task2_sft"
WANDB_RUN_TASK2_RL  = "a4_task2_rl"

_N_FINETUNE_GPUS = int(GPU_FINETUNE.split(":")[1]) if ":" in GPU_FINETUNE else 1

IDENTITY_JSONL_URL = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
)