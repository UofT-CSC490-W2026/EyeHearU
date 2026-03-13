"""
Part 3 Analysis: GSM8K problem clustering, reward/eval curves, and EDA.

Produces all plots and tables needed for A4 report sections 3.1 and 3.2.

Usage:
    1. Run the detailed eval on Modal first:
           cd a4/nanochat-modal
           uv run modal run nanochat_modal.py::stage_gsm8k_detailed_eval

    2. Download the JSON results:
           modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_sft.json data/gsm8k_detailed_sft.json
           modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl.json  data/gsm8k_detailed_rl.json

    3. Run this script:
           cd a4/part3
           python analysis.py

    Plots are saved to the plots/ directory.

Optional: Set WANDB_ENTITY and WANDB_RL_RUN_ID at the top to pull training
curves from W&B automatically. Otherwise the script uses the detailed eval
JSON files only.
"""

import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
PLOTS_DIR = "plots"
DATA_DIR = "data"
SFT_JSON = os.path.join(DATA_DIR, "gsm8k_detailed_sft.json")
RL_JSON = os.path.join(DATA_DIR, "gsm8k_detailed_rl.json")

# W&B settings (optional — set these to pull training curves automatically)
WANDB_ENTITY = None       # e.g. "maria-shurui-ma"
WANDB_PROJECT = "nanochat-rl"
WANDB_RL_RUN_ID = None    # e.g. "abc123xyz" — find this in the W&B URL

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


DOMAIN_KEYWORDS = {
    "money/shopping": ["dollar", "price", "cost", "pay", "earn", "spend",
                       "bought", "sold", "profit", "discount", "sale", "store",
                       "shop", "money", "wage", "salary", "budget", "cheap",
                       "expensive", "cent", "fee", "charge", "bill", "tax"],
    "time": ["hour", "minute", "second", "day", "week", "month", "year",
             "time", "clock", "schedule", "duration", "morning", "evening"],
    "food/cooking": ["recipe", "cook", "bake", "ingredient", "cake", "pie",
                     "cookie", "bread", "pizza", "chicken", "egg", "cup",
                     "tablespoon", "ounce", "pound", "gallon", "liter"],
    "distance/travel": ["mile", "kilometer", "drive", "walk", "run", "trip",
                        "travel", "speed", "distance", "road", "car", "bus"],
    "people/age": ["age", "old", "young", "birthday", "born", "people",
                   "friend", "family", "student", "class", "teacher"],
    "counting/inventory": ["many", "total", "number", "count", "remain",
                           "left", "collect", "gather", "box", "bag", "pack"],
}


def classify_domain(question):
    q_lower = question.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def classify_num_steps(result):
    return result.get("gt_num_tool_calls", 0)


def classify_answer_magnitude(result):
    ans = result.get("gt_answer")
    if ans is None:
        return "unknown"
    try:
        val = abs(float(ans))
    except (ValueError, TypeError):
        return "unknown"
    if val < 10:
        return "small (<10)"
    elif val < 100:
        return "medium (10-99)"
    elif val < 1000:
        return "large (100-999)"
    else:
        return "very large (1000+)"


def classify_question_length(question):
    n = len(question.split())
    if n < 30:
        return "short (<30 words)"
    elif n < 60:
        return "medium (30-59)"
    else:
        return "long (60+)"


OPERATION_PATTERNS = {
    "addition": re.compile(r"[\d.]+\s*\+\s*[\d.]+"),
    "subtraction": re.compile(r"[\d.]+\s*\-\s*[\d.]+"),
    "multiplication": re.compile(r"[\d.]+\s*\*\s*[\d.]+"),
    "division": re.compile(r"[\d.]+\s*/\s*[\d.]+"),
}


def classify_operations(result):
    """Return the set of arithmetic operations used in the ground truth."""
    tool_calls = result.get("gt_tool_calls", [])
    ops_found = set()
    for expr in tool_calls:
        for op_name, pattern in OPERATION_PATTERNS.items():
            if pattern.search(expr):
                ops_found.add(op_name)
    return ops_found if ops_found else {"unknown"}


def classify_error_type(result):
    """For incorrect answers, classify the type of error."""
    if result["is_correct"]:
        return "correct"
    pred = result.get("pred_answer")
    gt = result.get("gt_answer")
    response = result.get("model_response", "")

    if pred is None or "####" not in response:
        return "format_error"
    if not any(c in response for c in ["<<", "python", "calc"]):
        return "no_tool_use"
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) < 0.1:
            return "close_arithmetic"
        return "wrong_arithmetic"
    except (ValueError, TypeError):
        return "wrong_arithmetic"


# ─── Plot helpers ────────────────────────────────────────────────────────────

def bar_chart(categories, values, title, xlabel, ylabel, filename,
              color="#4C72B0", horizontal=False, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(categories))
    if horizontal:
        ax.barh(positions, values, color=color)
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(positions, values, color=color)
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def grouped_bar_chart(categories, values_a, values_b, label_a, label_b,
                      title, ylabel, filename, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, values_a, width, label=label_a, color="#4C72B0")
    ax.bar(x + width / 2, values_b, width, label=label_b, color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ─── W&B Training Curves (optional) ─────────────────────────────────────────

def plot_wandb_curves():
    """Pull training curves from W&B and create plots."""
    if not WANDB_ENTITY or not WANDB_RL_RUN_ID:
        print("\n[W&B] Skipping training curve plots (WANDB_ENTITY / WANDB_RL_RUN_ID not set).")
        print("       To enable: set them at the top of this script, or export from W&B UI.")
        return

    try:
        import wandb
    except ImportError:
        print("\n[W&B] wandb not installed. Install with: pip install wandb")
        return

    print("\n[W&B] Pulling training curves...")
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RL_RUN_ID}")
    history = run.history(samples=5000)

    # Reward curve
    reward_data = history[history["reward"].notna()][["step", "reward"]]
    if not reward_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(reward_data["step"], reward_data["reward"], color="#4C72B0", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("RL Training: Mean Reward over Steps")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "reward_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved reward_curve.png")

    # Pass@k curves
    pass_cols = [c for c in history.columns if c.startswith("pass@")]
    if pass_cols:
        pass_data = history[history[pass_cols[0]].notna()][["step"] + pass_cols]
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in sorted(pass_cols, key=lambda c: int(c.split("@")[1])):
            ax.plot(pass_data["step"], pass_data[col], label=col, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("RL Training: Pass@k on GSM8K Test Set")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "passk_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved passk_curves.png")

    # Sequence length
    seq_data = history[history["sequence_length"].notna()][["step", "sequence_length"]]
    if not seq_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(seq_data["step"], seq_data["sequence_length"], color="#55A868", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Sequence Length (tokens)")
        ax.set_title("RL Training: Average Generation Length")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "sequence_length.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved sequence_length.png")


# ─── Main Analysis ───────────────────────────────────────────────────────────

def analyze_one(data, label):
    """Analyze a single eval run, return category → accuracy dicts."""
    results = data["results"]
    total = data["total"]
    correct = data["correct"]
    print(f"\n{'='*60}")
    print(f"  {label}: {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"{'='*60}")

    analyses = {}

    # --- By domain ---
    domain_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        d = classify_domain(r["question"])
        domain_counts[d]["total"] += 1
        domain_counts[d]["correct"] += int(r["is_correct"])
    analyses["domain"] = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in sorted(domain_counts.items(), key=lambda x: -x[1]["total"])
    }
    analyses["domain_counts"] = {
        k: v["total"]
        for k, v in sorted(domain_counts.items(), key=lambda x: -x[1]["total"])
    }
    print(f"\n  By domain:")
    for d, acc in analyses["domain"].items():
        cnt = domain_counts[d]["total"]
        print(f"    {d:25s}  {100*acc:5.1f}%  (n={cnt})")

    # --- By number of reasoning steps ---
    step_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = classify_num_steps(r)
        step_counts[s]["total"] += 1
        step_counts[s]["correct"] += int(r["is_correct"])
    analyses["steps"] = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in sorted(step_counts.items())
    }
    analyses["steps_counts"] = {
        k: v["total"] for k, v in sorted(step_counts.items())
    }
    print(f"\n  By number of reasoning steps (tool calls):")
    for s, acc in analyses["steps"].items():
        cnt = step_counts[s]["total"]
        print(f"    {s} steps:  {100*acc:5.1f}%  (n={cnt})")

    # --- By answer magnitude ---
    mag_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        m = classify_answer_magnitude(r)
        mag_counts[m]["total"] += 1
        mag_counts[m]["correct"] += int(r["is_correct"])
    mag_order = ["small (<10)", "medium (10-99)", "large (100-999)",
                 "very large (1000+)", "unknown"]
    analyses["magnitude"] = {
        k: mag_counts[k]["correct"] / mag_counts[k]["total"]
        if mag_counts[k]["total"] > 0 else 0
        for k in mag_order if mag_counts[k]["total"] > 0
    }
    print(f"\n  By answer magnitude:")
    for m in mag_order:
        if mag_counts[m]["total"] > 0:
            acc = mag_counts[m]["correct"] / mag_counts[m]["total"]
            print(f"    {m:25s}  {100*acc:5.1f}%  (n={mag_counts[m]['total']})")

    # --- By question length ---
    len_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        l = classify_question_length(r["question"])
        len_counts[l]["total"] += 1
        len_counts[l]["correct"] += int(r["is_correct"])
    len_order = ["short (<30 words)", "medium (30-59)", "long (60+)"]
    analyses["question_length"] = {
        k: len_counts[k]["correct"] / len_counts[k]["total"]
        if len_counts[k]["total"] > 0 else 0
        for k in len_order if len_counts[k]["total"] > 0
    }
    print(f"\n  By question length:")
    for l in len_order:
        if len_counts[l]["total"] > 0:
            acc = len_counts[l]["correct"] / len_counts[l]["total"]
            print(f"    {l:25s}  {100*acc:5.1f}%  (n={len_counts[l]['total']})")

    # --- By operation type ---
    op_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        ops = classify_operations(r)
        for op in ops:
            op_counts[op]["total"] += 1
            op_counts[op]["correct"] += int(r["is_correct"])
    analyses["operations"] = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in sorted(op_counts.items(), key=lambda x: -x[1]["total"])
    }
    print(f"\n  By operation type (a problem can have multiple):")
    for op, acc in analyses["operations"].items():
        cnt = op_counts[op]["total"]
        print(f"    {op:20s}  {100*acc:5.1f}%  (n={cnt})")

    # --- Error type breakdown (incorrect only) ---
    error_counts = Counter()
    for r in results:
        etype = classify_error_type(r)
        if etype != "correct":
            error_counts[etype] += 1
    analyses["error_types"] = dict(error_counts.most_common())
    print(f"\n  Error type breakdown ({total - correct} errors):")
    for etype, cnt in error_counts.most_common():
        print(f"    {etype:25s}  {cnt:4d}  ({100*cnt/(total-correct):.1f}%)")

    return analyses


def main():
    # Check data files exist
    has_sft = os.path.exists(SFT_JSON)
    has_rl = os.path.exists(RL_JSON)

    if not has_sft and not has_rl:
        print("ERROR: No data files found in data/ directory.")
        print("Run these commands first:")
        print("  cd a4/nanochat-modal")
        print("  uv run modal run nanochat_modal.py::stage_gsm8k_detailed_eval")
        print("Then download:")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_sft.json ../part3/data/gsm8k_detailed_sft.json")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl.json  ../part3/data/gsm8k_detailed_rl.json")
        return

    sft_analyses = None
    rl_analyses = None

    if has_sft:
        sft_data = load_json(SFT_JSON)
        sft_analyses = analyze_one(sft_data, "After SFT")

    if has_rl:
        rl_data = load_json(RL_JSON)
        rl_analyses = analyze_one(rl_data, "After RL")

    # ─── Generate plots ──────────────────────────────────────────────────

    print(f"\nGenerating plots...")

    # Use RL data as primary (it's the Part 3 focus); SFT for comparison
    primary_data = rl_data if has_rl else sft_data
    primary_analyses = rl_analyses if has_rl else sft_analyses
    primary_label = "After RL" if has_rl else "After SFT"

    # 1. Accuracy by domain
    domains = list(primary_analyses["domain"].keys())
    domain_accs = [primary_analyses["domain"][d] * 100 for d in domains]
    domain_labels = [f"{d}\n(n={primary_analyses['domain_counts'][d]})" for d in domains]
    bar_chart(domain_labels, domain_accs,
              f"GSM8K Accuracy by Problem Domain ({primary_label})",
              "Domain", "Accuracy (%)",
              "accuracy_by_domain.png", figsize=(12, 5))

    # 2. Accuracy by number of reasoning steps
    steps = sorted(primary_analyses["steps"].keys())
    step_accs = [primary_analyses["steps"][s] * 100 for s in steps]
    step_labels = [str(s) for s in steps]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, step_accs, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Reasoning Steps (tool calls)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"GSM8K Accuracy vs. Reasoning Steps ({primary_label})")
    ax.grid(True, alpha=0.3)
    for s, acc in zip(steps, step_accs):
        cnt = primary_analyses["steps_counts"][s]
        ax.annotate(f"n={cnt}", (s, acc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "accuracy_by_steps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved accuracy_by_steps.png")

    # 3. Accuracy by answer magnitude
    mag_order = [m for m in ["small (<10)", "medium (10-99)",
                             "large (100-999)", "very large (1000+)"]
                 if m in primary_analyses["magnitude"]]
    mag_accs = [primary_analyses["magnitude"][m] * 100 for m in mag_order]
    bar_chart(mag_order, mag_accs,
              f"GSM8K Accuracy by Answer Magnitude ({primary_label})",
              "Answer Magnitude", "Accuracy (%)",
              "accuracy_by_magnitude.png")

    # 4. Accuracy by operation type
    ops = list(primary_analyses["operations"].keys())
    op_accs = [primary_analyses["operations"][o] * 100 for o in ops]
    bar_chart(ops, op_accs,
              f"GSM8K Accuracy by Operation Type ({primary_label})",
              "Operation", "Accuracy (%)",
              "accuracy_by_operation.png")

    # 5. Error type breakdown (pie chart)
    if primary_analyses["error_types"]:
        etypes = list(primary_analyses["error_types"].keys())
        ecounts = list(primary_analyses["error_types"].values())
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["#C44E52", "#DD8452", "#CCB974", "#8172B3", "#64B5CD"]
        ax.pie(ecounts, labels=etypes, autopct="%1.1f%%",
               colors=colors[:len(etypes)], startangle=90)
        ax.set_title(f"Error Type Distribution ({primary_label})")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "error_types.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved error_types.png")

    # 6. SFT vs RL comparison (if both available)
    if has_sft and has_rl:
        # By domain
        common_domains = [d for d in domains if d in sft_analyses["domain"]]
        sft_domain_accs = [sft_analyses["domain"].get(d, 0) * 100 for d in common_domains]
        rl_domain_accs = [rl_analyses["domain"].get(d, 0) * 100 for d in common_domains]
        grouped_bar_chart(common_domains, sft_domain_accs, rl_domain_accs,
                          "After SFT", "After RL",
                          "GSM8K Accuracy by Domain: SFT vs RL",
                          "Accuracy (%)", "sft_vs_rl_domain.png", figsize=(12, 5))

        # By steps
        common_steps = sorted(set(sft_analyses["steps"].keys()) |
                              set(rl_analyses["steps"].keys()))
        sft_step_accs = [sft_analyses["steps"].get(s, 0) * 100 for s in common_steps]
        rl_step_accs = [rl_analyses["steps"].get(s, 0) * 100 for s in common_steps]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(common_steps, sft_step_accs, "o-", label="After SFT",
                color="#4C72B0", linewidth=2, markersize=8)
        ax.plot(common_steps, rl_step_accs, "s-", label="After RL",
                color="#DD8452", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Reasoning Steps")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("GSM8K Accuracy vs. Reasoning Steps: SFT vs RL")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "sft_vs_rl_steps.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved sft_vs_rl_steps.png")

        # Error type comparison
        all_etypes = sorted(set(sft_analyses["error_types"].keys()) |
                            set(rl_analyses["error_types"].keys()))
        sft_ecounts = [sft_analyses["error_types"].get(e, 0) for e in all_etypes]
        rl_ecounts = [rl_analyses["error_types"].get(e, 0) for e in all_etypes]
        grouped_bar_chart(all_etypes, sft_ecounts, rl_ecounts,
                          "After SFT", "After RL",
                          "Error Type Distribution: SFT vs RL",
                          "Count", "sft_vs_rl_errors.png")

    # ─── W&B curves ──────────────────────────────────────────────────────
    plot_wandb_curves()

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  All plots saved to {PLOTS_DIR}/")
    print(f"{'='*60}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".png"):
            print(f"  - {PLOTS_DIR}/{f}")


if __name__ == "__main__":
    main()
