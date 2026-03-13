"""
Part 4 Analysis: Comparison of reward configurations for RL on GSM8K.

Generates bar charts and heatmaps comparing benchmark results across:
  - Baseline RL (correctness only)
  - Combined rewards (correctness + format + steps + close)
  - Format-only (correctness + format)
  - Close-only (correctness + close)

Also performs per-problem error analysis if detailed JSON files are available.

Usage:
    cd a4/part4
    python analysis.py

    (Optional) To include per-problem error analysis, first download JSON files:
        modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl.json        data/gsm8k_detailed_rl_baseline.json
        modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_combined.json data/gsm8k_detailed_rl_combined.json
        modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_format.json   data/gsm8k_detailed_rl_format.json
        modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_close.json    data/gsm8k_detailed_rl_close.json

Plots are saved to plots/
"""

import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "plots"
DATA_DIR = "data"
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIGS = {
    "After SFT":     {"ARC-Easy": 36.20, "ARC-Challenge": 32.85, "MMLU": 30.71, "GSM8K": 3.56,  "HumanEval": 6.71, "SpellingBee": 99.22, "ChatCORE": 0.2375},
    "Baseline RL":   {"ARC-Easy": 35.90, "ARC-Challenge": 30.46, "MMLU": 31.04, "GSM8K": 10.92, "HumanEval": 0.00, "SpellingBee": 2.73,  "ChatCORE": 0.0725},
    "Combined":      {"ARC-Easy": 34.93, "ARC-Challenge": 32.59, "MMLU": 30.64, "GSM8K": 10.24, "HumanEval": 0.00, "SpellingBee": 32.42, "ChatCORE": 0.1226},
    "Format-Only":   {"ARC-Easy": 33.33, "ARC-Challenge": 32.59, "MMLU": 29.30, "GSM8K": 7.35,  "HumanEval": 0.61, "SpellingBee": 0.00,  "ChatCORE": 0.0582},
    "Close-Only":    {"ARC-Easy": 35.10, "ARC-Challenge": 29.44, "MMLU": 31.07, "GSM8K": 10.92, "HumanEval": 1.22, "SpellingBee": 91.41, "ChatCORE": 0.2184},
}

COLORS = {
    "After SFT":   "#7FB3D8",
    "Baseline RL": "#4C72B0",
    "Combined":    "#55A868",
    "Format-Only": "#C44E52",
    "Close-Only":  "#DD8452",
}

RL_CONFIGS = ["Baseline RL", "Combined", "Format-Only", "Close-Only"]
ALL_CONFIGS = list(CONFIGS.keys())
BENCHMARKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]


def plot_single_benchmark(benchmark, filename, figsize=(8, 5)):
    """Bar chart for a single benchmark across all configurations."""
    fig, ax = plt.subplots(figsize=figsize)
    names = ALL_CONFIGS
    values = [CONFIGS[c][benchmark] for c in names]
    colors = [COLORS[c] for c in names]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{benchmark} Accuracy Across Configurations")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_chatcore(filename="chatcore_comparison.png"):
    """Bar chart for ChatCORE metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ALL_CONFIGS
    values = [CONFIGS[c]["ChatCORE"] for c in names]
    colors = [COLORS[c] for c in names]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ChatCORE")
    ax.set_title("ChatCORE Metric Across Configurations")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_benchmark_heatmap(filename="benchmark_heatmap.png"):
    """Heatmap of all benchmarks vs. all configurations."""
    data = np.array([[CONFIGS[c][b] for b in BENCHMARKS] for c in ALL_CONFIGS])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels(BENCHMARKS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ALL_CONFIGS)))
    ax.set_yticklabels(ALL_CONFIGS, fontsize=9)

    for i in range(len(ALL_CONFIGS)):
        for j in range(len(BENCHMARKS)):
            val = data[i, j]
            color = "white" if val < 20 or val > 80 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_title("Benchmark Accuracy Heatmap: All Configurations")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_grouped_benchmarks(filename="grouped_benchmarks.png"):
    """Grouped bar chart of all benchmarks for RL configurations only."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BENCHMARKS))
    n = len(RL_CONFIGS)
    width = 0.8 / n

    for i, config in enumerate(RL_CONFIGS):
        values = [CONFIGS[config][b] for b in BENCHMARKS]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=config,
                       color=COLORS[config], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Comparison: RL Reward Configurations")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ─── Per-problem error analysis (requires downloaded JSON files) ─────────────

def classify_error_type(result):
    """Classify the error type for an incorrect prediction."""
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


def analyze_errors(data, label):
    """Analyze error distribution for a single model."""
    results = data["results"]
    errors = [r for r in results if not r["is_correct"]]
    error_counts = Counter(classify_error_type(r) for r in errors)
    total_errors = len(errors)
    print(f"\n  {label}: {data['correct']}/{data['total']} correct "
          f"({100*data['accuracy']:.2f}%), {total_errors} errors")
    for etype, cnt in error_counts.most_common():
        print(f"    {etype:25s}  {cnt:4d}  ({100*cnt/total_errors:.1f}%)")
    return dict(error_counts)


def plot_error_comparison(all_errors, filename="error_type_comparison.png"):
    """Grouped bar chart comparing error types across models."""
    all_etypes = sorted(set().union(*[set(e.keys()) for e in all_errors.values()]))
    configs = list(all_errors.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_etypes))
    n = len(configs)
    width = 0.8 / n

    config_colors = {
        "Baseline RL": COLORS["Baseline RL"],
        "Combined": COLORS["Combined"],
        "Format-Only": COLORS["Format-Only"],
        "Close-Only": COLORS["Close-Only"],
    }

    for i, config in enumerate(configs):
        values = [all_errors[config].get(e, 0) for e in all_etypes]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=config,
               color=config_colors.get(config, f"C{i}"),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_etypes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Error Count")
    ax.set_title("Error Type Distribution: All RL Configurations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_per_problem_comparison(data_baseline, data_close, filename="per_problem_diff.png"):
    """Scatter plot showing per-problem correctness difference between baseline and close-only."""
    baseline_map = {r["idx"]: r["is_correct"] for r in data_baseline["results"]}
    close_map = {r["idx"]: r["is_correct"] for r in data_close["results"]}

    common_idx = sorted(set(baseline_map.keys()) & set(close_map.keys()))

    categories = {"both_correct": 0, "both_wrong": 0,
                  "baseline_only": 0, "close_only": 0}
    for idx in common_idx:
        bc = baseline_map[idx]
        cc = close_map[idx]
        if bc and cc:
            categories["both_correct"] += 1
        elif not bc and not cc:
            categories["both_wrong"] += 1
        elif bc and not cc:
            categories["baseline_only"] += 1
        else:
            categories["close_only"] += 1

    labels = ["Both Correct", "Both Wrong", "Baseline Only", "Close Only"]
    values = [categories["both_correct"], categories["both_wrong"],
              categories["baseline_only"], categories["close_only"]]
    colors_pie = ["#55A868", "#C44E52", "#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(values, labels=[f"{l}\n(n={v})" for l, v in zip(labels, values)],
           autopct="%1.1f%%", colors=colors_pie, startangle=90)
    ax.set_title("Per-Problem Agreement: Baseline RL vs. Close-Only RL")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def main():
    print("=" * 60)
    print("  Part 4 Analysis: Reward Configuration Comparison")
    print("=" * 60)

    print("\nGenerating aggregate benchmark plots...")
    plot_single_benchmark("GSM8K", "gsm8k_comparison.png")
    plot_single_benchmark("SpellingBee", "spellingbee_comparison.png")
    plot_chatcore()
    plot_benchmark_heatmap()
    plot_grouped_benchmarks()

    json_files = {
        "Baseline RL": os.path.join(DATA_DIR, "gsm8k_detailed_rl_baseline.json"),
        "Combined":    os.path.join(DATA_DIR, "gsm8k_detailed_rl_combined.json"),
        "Format-Only": os.path.join(DATA_DIR, "gsm8k_detailed_rl_format.json"),
        "Close-Only":  os.path.join(DATA_DIR, "gsm8k_detailed_rl_close.json"),
    }

    available = {k: v for k, v in json_files.items() if os.path.exists(v)}

    if available:
        print(f"\nFound {len(available)} detailed eval JSON files. Running error analysis...")
        all_errors = {}
        all_data = {}
        for config, path in available.items():
            with open(path) as f:
                data = json.load(f)
            all_data[config] = data
            all_errors[config] = analyze_errors(data, config)

        print("\nGenerating error analysis plots...")
        plot_error_comparison(all_errors)

        if "Baseline RL" in all_data and "Close-Only" in all_data:
            plot_per_problem_comparison(all_data["Baseline RL"], all_data["Close-Only"])
    else:
        print("\nNo detailed eval JSON files found in data/.")
        print("To enable per-problem error analysis, download them from Modal:")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl.json        data/gsm8k_detailed_rl_baseline.json")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_combined.json data/gsm8k_detailed_rl_combined.json")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_format.json   data/gsm8k_detailed_rl_format.json")
        print("  modal volume get nanochat-vol nanochat_cache/report/gsm8k_detailed_rl_close.json    data/gsm8k_detailed_rl_close.json")

    print(f"\n{'=' * 60}")
    print(f"  All plots saved to {PLOTS_DIR}/")
    print(f"{'=' * 60}")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".png"):
            print(f"  - {PLOTS_DIR}/{f}")


if __name__ == "__main__":
    main()
