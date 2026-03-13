"""
Part 3: RL Run Analysis — reward curves, eval curves, and GSM8K problem clustering.

Usage:
    cd a4/part3
    pip install matplotlib pandas datasets
    python analysis.py
"""
import re
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TERMINAL_LOG = os.path.expanduser(
    "~/.cursor/projects/Users-chloe-csc490-EyeHearU/terminals/559760.txt"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 1. Extract step-level reward data from terminal log ─────────────────────

def extract_step_rewards(log_path):
    pattern = re.compile(
        r"Step (\d+)/467 \| Average reward: ([\d.]+) \| Average sequence length: ([\d.]+)"
    )
    rows = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "reward": float(m.group(2)),
                    "seq_len": float(m.group(3)),
                })
    return pd.DataFrame(rows)

# ─── 2. Plot reward curve ────────────────────────────────────────────────────

def plot_reward_curve(df):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    window = 20
    df["reward_smooth"] = df["reward"].rolling(window, min_periods=1).mean()

    ax1.plot(df["step"], df["reward"], alpha=0.2, color="blue", label="Raw reward")
    ax1.plot(df["step"], df["reward_smooth"], color="blue", linewidth=2, label=f"Smoothed (window={window})")
    ax1.set_xlabel("RL Step")
    ax1.set_ylabel("Average Reward", color="blue")
    ax1.set_title("RL Training: Reward Curve (GRPO on GSM8K)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["seq_len"], alpha=0.4, color="red", linewidth=1)
    ax2.set_ylabel("Average Sequence Length", color="red")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

# ─── 3. Plot pass@k eval curve ──────────────────────────────────────────────

def plot_passk_curve():
    """
    Reconstruct pass@k from W&B sparkline: eval every 60 steps.
    W&B summary: pass@1 ▂▇█▇▇▆▂▁  (8 eval points)
    Final: pass@1=0.0125, pass@8=0.0425
    """
    eval_steps = [60, 120, 180, 240, 300, 360, 420, 466]

    sparkline_map = {"▁": 1, "▂": 2, "▃": 3, "▄": 4, "▅": 5, "▆": 6, "▇": 7, "█": 8}

    def decode_sparkline(spark, final_val):
        vals = [sparkline_map.get(c, 0) for c in spark]
        max_v = max(vals) if vals else 1
        return [v / max_v * (final_val / (vals[-1] / max_v)) if max_v else 0 for v in vals]

    pass1_spark = "▂▇█▇▇▆▂▁"
    pass8_spark = "▄▇█▇█▆▁▁"

    pass1_final = 0.0125
    pass8_final = 0.0425

    pass1_raw = [sparkline_map.get(c, 0) for c in pass1_spark]
    pass8_raw = [sparkline_map.get(c, 0) for c in pass8_spark]

    pass1_max_idx = pass1_raw.index(max(pass1_raw))
    pass8_max_idx = pass8_raw.index(max(pass8_raw))

    pass1_peak = pass1_final * (max(pass1_raw) / pass1_raw[-1]) if pass1_raw[-1] else 0.05
    pass8_peak = pass8_final * (max(pass8_raw) / pass8_raw[-1]) if pass8_raw[-1] else 0.15

    pass1_values = [v / max(pass1_raw) * pass1_peak for v in pass1_raw]
    pass8_values = [v / max(pass8_raw) * pass8_peak for v in pass8_raw]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, pass1_values, "o-", color="blue", linewidth=2, markersize=6, label="pass@1")
    ax.plot(eval_steps, pass8_values, "s-", color="green", linewidth=2, markersize=6, label="pass@8")

    ax.axhline(y=0.0311, color="gray", linestyle="--", alpha=0.6, label="SFT baseline (GSM8K 3.11%)")

    ax.set_xlabel("RL Step")
    ax.set_ylabel("GSM8K Accuracy")
    ax.set_title("RL Training: GSM8K pass@k Eval Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "passk_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return eval_steps, pass1_values, pass8_values

# ─── 4. Benchmark comparison chart ──────────────────────────────────────────

def plot_benchmark_comparison():
    tasks = ["ARC-Easy", "ARC-Chall", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    pretrained = [25, 25, 25, 0, 0, 0]
    after_sft = [36.15, 30.12, 31.39, 3.11, 8.54, 98.44]
    after_rl = [32.07, 31.48, 28.79, 4.32, 0.0, 0.0]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, pretrained, width, label="Pretrained", color="#9ecae1", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, after_sft, width, label="After SFT", color="#4292c6", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, after_rl, width, label="After RL", color="#08519c", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Comparison: Pretrained → SFT → RL")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "benchmark_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

# ─── 5. GSM8K problem analysis and clustering ───────────────────────────────

def analyze_gsm8k_problems():
    """Categorize GSM8K problems by mathematical operation type and difficulty."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        print("Could not load GSM8K dataset, using manual categorization")
        return None

    categories = {
        "arithmetic": [],
        "multi_step": [],
        "fractions_ratios": [],
        "money_shopping": [],
        "time_scheduling": [],
        "geometry_measurement": [],
        "comparison": [],
        "percentage": [],
        "other": [],
    }

    keywords = {
        "money_shopping": ["dollar", "cost", "price", "buy", "sell", "pay", "earn", "spend", "profit", "store", "shop"],
        "time_scheduling": ["hour", "minute", "day", "week", "month", "year", "time", "schedule", "clock"],
        "fractions_ratios": ["half", "third", "quarter", "fraction", "ratio", "twice", "triple", "double"],
        "percentage": ["percent", "%"],
        "geometry_measurement": ["mile", "meter", "foot", "inch", "area", "length", "width", "height", "distance"],
        "comparison": ["more than", "less than", "fewer", "greater", "difference", "compare"],
    }

    for i, example in enumerate(ds):
        question = example["question"].lower()
        answer = example["answer"]
        num_steps = answer.count("<<")

        categorized = False
        for cat, kws in keywords.items():
            if any(kw in question for kw in kws):
                categories[cat].append({"idx": i, "question": example["question"], "steps": num_steps})
                categorized = True
                break

        if not categorized:
            if num_steps >= 4:
                categories["multi_step"].append({"idx": i, "question": example["question"], "steps": num_steps})
            elif num_steps <= 1:
                categories["arithmetic"].append({"idx": i, "question": example["question"], "steps": num_steps})
            else:
                categories["other"].append({"idx": i, "question": example["question"], "steps": num_steps})

    cat_counts = {k: len(v) for k, v in categories.items()}
    avg_steps = {k: np.mean([p["steps"] for p in v]) if v else 0 for k, v in categories.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cats = sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True)
    counts = [cat_counts[c] for c in cats]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))

    ax1.barh(cats, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Number of Problems")
    ax1.set_title("GSM8K Test Set: Problem Categories")
    for i, v in enumerate(counts):
        ax1.text(v + 5, i, str(v), va="center", fontsize=9)

    steps = [avg_steps[c] for c in cats]
    ax2.barh(cats, steps, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Average Reasoning Steps")
    ax2.set_title("GSM8K: Avg Steps per Category")
    for i, v in enumerate(steps):
        ax2.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gsm8k_categories.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    step_dist = [example["answer"].count("<<") for example in ds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(step_dist, bins=range(0, max(step_dist) + 2), color="#4292c6", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Number of Reasoning Steps (<<...>> operations)")
    ax.set_ylabel("Number of Problems")
    ax.set_title("GSM8K Test Set: Distribution of Reasoning Complexity")
    ax.axvline(x=np.mean(step_dist), color="red", linestyle="--", label=f"Mean = {np.mean(step_dist):.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gsm8k_step_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return cat_counts, avg_steps

# ─── 6. Reward distribution analysis ────────────────────────────────────────

def plot_reward_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(df["reward"], bins=30, color="#4292c6", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Step Reward")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Per-Step Rewards")
    ax1.axvline(x=df["reward"].mean(), color="red", linestyle="--", label=f"Mean = {df['reward'].mean():.4f}")
    ax1.legend()

    thirds = len(df) // 3
    early = df.iloc[:thirds]["reward"]
    mid = df.iloc[thirds:2*thirds]["reward"]
    late = df.iloc[2*thirds:]["reward"]

    ax2.boxplot([early, mid, late], labels=["Early\n(0-155)", "Mid\n(156-310)", "Late\n(311-466)"])
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward Distribution by Training Phase")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "reward_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Part 3: RL Run Analysis")
    print("=" * 60)

    print("\n[1] Extracting step rewards from terminal log...")
    df = extract_step_rewards(TERMINAL_LOG)
    print(f"    Extracted {len(df)} steps")
    print(f"    Reward range: {df['reward'].min():.4f} - {df['reward'].max():.4f}")
    print(f"    Mean reward: {df['reward'].mean():.4f}")
    print(f"    Seq length range: {df['seq_len'].min():.1f} - {df['seq_len'].max():.1f}")

    print("\n[2] Plotting reward curve...")
    plot_reward_curve(df)

    print("\n[3] Plotting pass@k eval curve...")
    eval_steps, pass1, pass8 = plot_passk_curve()
    print(f"    pass@1 values: {[f'{v:.4f}' for v in pass1]}")
    print(f"    pass@8 values: {[f'{v:.4f}' for v in pass8]}")

    print("\n[4] Plotting benchmark comparison...")
    plot_benchmark_comparison()

    print("\n[5] Plotting reward distribution...")
    plot_reward_distribution(df)

    print("\n[6] Analyzing GSM8K problem categories...")
    result = analyze_gsm8k_problems()
    if result:
        cat_counts, avg_steps = result
        print("    Categories:")
        for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"      {k}: {v} problems (avg {avg_steps[k]:.1f} steps)")

    print("\n" + "=" * 60)
    print("All plots saved to:", OUTPUT_DIR)
    print("=" * 60)
