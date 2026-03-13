import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent          # a4/part4
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "baseline": DATA_DIR / "gsm8k_detailed_rl_baseline.json",
    "format":   DATA_DIR / "gsm8k_detailed_rl_format.json",
    "close":    DATA_DIR / "gsm8k_detailed_rl_close.json",
    "combined": DATA_DIR / "gsm8k_detailed_rl_combined.json",
}

def load_results(path: Path):
    with path.open() as f:
        obj = json.load(f)
    return obj["results"]


# --- Classification helpers (mirroring Part 3) --------------------------------

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


def classify_domain(question: str) -> str:
    q_lower = question.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def classify_num_steps(result) -> int:
    # Ground-truth number of calculator/tool calls, like Part 3
    return result.get("gt_num_tool_calls", 0)


def classify_answer_magnitude(result) -> str:
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


def classify_operations(result):
    """Rudimentary operation classification using tool-call strings."""
    tool_calls = result.get("gt_tool_calls", [])
    ops_found = set()
    for expr in tool_calls:
        if "*" in expr:
            ops_found.add("multiplication")
        if "+" in expr:
            ops_found.add("addition")
        if "-" in expr:
            ops_found.add("subtraction")
        if "/" in expr:
            ops_found.add("division")
    return ops_found if ops_found else {"unknown"}

def classify_error(gt_answer: str, pred_answer, model_response: str):
    """
    Rebuild the simple Part 3 taxonomy for GSM8K only.
    Assumes gt_answer is a string like '109' and pred_answer is either a string or None.
    """
    if pred_answer is None:
        # could not extract a number at all
        return "format_error"

    # format looks ok; now check arithmetic distance
    try:
        gt = float(gt_answer)
        pred = float(pred_answer)
    except Exception:
        return "format_error"

    if gt == pred:
        return "correct"

    # relative error if gt not tiny, else absolute
    diff = abs(gt - pred)
    if abs(gt) > 1:
        rel = diff / abs(gt)
        if rel <= 0.10:
            return "close_arithmetic"
    else:
        if diff <= 0.5:
            return "close_arithmetic"

    return "wrong_arithmetic"

def compute_stats(results):
    n = len(results)
    correct = sum(r["is_correct"] for r in results)
    acc = correct / n

    errors = Counter()
    for r in results:
        etype = classify_error(r["gt_answer"], r["pred_answer"], r["model_response"])
        errors[etype] += 1
    return acc, errors


def analyze_model(results):
    """Compute per-category accuracies for one configuration (GSM8K only)."""
    domain_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    step_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    mag_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    op_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        is_corr = int(r["is_correct"])

        d = classify_domain(r["question"])
        domain_counts[d]["total"] += 1
        domain_counts[d]["correct"] += is_corr

        s = classify_num_steps(r)
        step_counts[s]["total"] += 1
        step_counts[s]["correct"] += is_corr

        m = classify_answer_magnitude(r)
        mag_counts[m]["total"] += 1
        mag_counts[m]["correct"] += is_corr

        for op in classify_operations(r):
            op_counts[op]["total"] += 1
            op_counts[op]["correct"] += is_corr

    def to_acc(counts_dict):
        return {k: (v["correct"] / v["total"] if v["total"] > 0 else 0.0)
                for k, v in counts_dict.items()}

    return {
        "domain_acc": to_acc(domain_counts),
        "domain_total": {k: v["total"] for k, v in domain_counts.items()},
        "steps_acc": to_acc(step_counts),
        "steps_total": {k: v["total"] for k, v in step_counts.items()},
        "mag_acc": to_acc(mag_counts),
        "mag_total": {k: v["total"] for k, v in mag_counts.items()},
        "ops_acc": to_acc(op_counts),
        "ops_total": {k: v["total"] for k, v in op_counts.items()},
    }

def plot_gsm8k_accuracy(acc_by_model, out_path=PLOTS_DIR/ "gsm8k_comparison.png"):
    names = list(acc_by_model.keys())
    vals = [100 * acc_by_model[n] for n in names]

    plt.figure(figsize=(6, 4))
    plt.bar(names, vals, color="#4c72b0")
    plt.ylabel("GSM8K accuracy (%)")
    plt.title("GSM8K Accuracy Across RL Reward Configurations")
    plt.ylim(0, max(vals) + 5)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_error_types(error_by_model, out_path=PLOTS_DIR / "error_type_comparison.png"):
    # models × error_types → grouped bar chart
    models = list(error_by_model.keys())
    error_types = ["wrong_arithmetic", "close_arithmetic", "format_error"]
    x = np.arange(len(error_types))
    width = 0.18

    plt.figure(figsize=(8, 4))
    for i, m in enumerate(models):
        counts = [error_by_model[m].get(t, 0) for t in error_types]
        plt.bar(x + i * width, counts, width=width, label=m)

    plt.xticks(x + width * (len(models) - 1) / 2, error_types, rotation=15)
    plt.ylabel("Count")
    plt.title("GSM8K Error Types by Reward Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_domain_by_config(analyses, out_path=PLOTS_DIR / "domain_by_config.png"):
    # Domains ordered by total support across configs (like Part 3)
    all_domains = sorted(
        set().union(*(a["domain_acc"].keys() for a in analyses.values())),
        key=lambda d: -sum(a["domain_total"].get(d, 0) for a in analyses.values()),
    )
    x = np.arange(len(all_domains))
    width = 0.18

    plt.figure(figsize=(10, 4))
    for i, (name, a) in enumerate(analyses.items()):
        vals = [a["domain_acc"].get(d, 0.0) * 100 for d in all_domains]
        plt.bar(x + i * width, vals, width=width, label=name)
    labels = [f"{d}\n(n={sum(a['domain_total'].get(d, 0) for a in analyses.values())})"
              for d in all_domains]
    plt.xticks(x + width * (len(analyses) - 1) / 2, labels, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("GSM8K Accuracy by Domain (All RL Configurations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_magnitude_by_config(analyses, out_path=PLOTS_DIR / "magnitude_by_config.png"):
    order = ["small (<10)", "medium (10-99)", "large (100-999)", "very large (1000+)", "unknown"]
    present = [m for m in order if any(m in a["mag_acc"] for a in analyses.values())]
    x = np.arange(len(present))
    width = 0.18

    plt.figure(figsize=(8, 4))
    for i, (name, a) in enumerate(analyses.items()):
        vals = [a["mag_acc"].get(m, 0.0) * 100 for m in present]
        plt.bar(x + i * width, vals, width=width, label=name)
    plt.xticks(x + width * (len(analyses) - 1) / 2, present, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("GSM8K Accuracy by Answer Magnitude (All RL Configurations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_operations_by_config(analyses, out_path=PLOTS_DIR / "operations_by_config.png"):
    all_ops = sorted(set().union(*(a["ops_acc"].keys() for a in analyses.values())),
                     key=lambda o: -sum(a["ops_total"].get(o, 0) for a in analyses.values()))
    x = np.arange(len(all_ops))
    width = 0.18

    plt.figure(figsize=(8, 4))
    for i, (name, a) in enumerate(analyses.items()):
        vals = [a["ops_acc"].get(o, 0.0) * 100 for o in all_ops]
        plt.bar(x + i * width, vals, width=width, label=name)
    plt.xticks(x + width * (len(analyses) - 1) / 2, all_ops, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("GSM8K Accuracy by Operation Type (All RL Configurations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_steps_by_config(analyses, out_path=PLOTS_DIR / "steps_by_config.png"):
    all_steps = sorted(set().union(*(a["steps_acc"].keys() for a in analyses.values())))
    plt.figure(figsize=(8, 4))
    for name, a in analyses.items():
        ys = [a["steps_acc"].get(s, 0.0) * 100 for s in all_steps]
        plt.plot(all_steps, ys, marker="o", label=name)
    plt.xlabel("Number of reasoning steps (tool calls)")
    plt.ylabel("Accuracy (%)")
    plt.title("GSM8K Accuracy vs. Reasoning Steps (All RL Configurations)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_per_problem_diff(baseline_results, other_results, label, out_path=PLOTS_DIR / "per_problem_diff.png"):
    """
    Simple scatter: per‑problem correctness difference baseline vs another model.
    """
    # assume results share the same ordering / idx
    base_correct = np.array([int(r["is_correct"]) for r in baseline_results])
    other_correct = np.array([int(r["is_correct"]) for r in other_results])

    diff = other_correct - base_correct  # -1, 0, +1

    plt.figure(figsize=(8, 2.5))
    plt.scatter(range(len(diff)), diff, s=5, c=np.where(diff > 0, "#2ca02c",
                                                       np.where(diff < 0, "#d62728", "#7f7f7f")))
    plt.yticks([-1, 0, 1], ["worse", "same", "better"])
    plt.xlabel("GSM8K problem index")
    plt.title(f"Per‑problem change vs baseline RL ({label})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    results = {name: load_results(path) for name, path in FILES.items()}

    acc_by_model = {}
    error_by_model = {}
    analyses_by_model = {}
    for name, rs in results.items():
        acc, errors = compute_stats(rs)
        acc_by_model[name] = acc
        error_by_model[name] = errors
        analyses_by_model[name] = analyze_model(rs)

    # Global accuracy and error-type plots
    plot_gsm8k_accuracy(acc_by_model)
    plot_error_types(error_by_model)

    # Part-3-like breakdowns (domain, steps, magnitude, operations)
    plot_domain_by_config(analyses_by_model)
    plot_steps_by_config(analyses_by_model)
    plot_magnitude_by_config(analyses_by_model)
    plot_operations_by_config(analyses_by_model)

    # Per-problem correctness difference: baseline vs close-only (math only)
    plot_per_problem_diff(
        baseline_results=results["baseline"],
        other_results=results["close"],
        label="close‑only",
    )

if __name__ == "__main__":
    main()