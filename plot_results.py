"""
plot_results.py — Generate evaluation graphs from evaluation_results.csv
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "evaluation_results.csv"


def load_results(filepath):
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows):
    strategies = ["zero-shot", "few-shot", "chain-of-thought"]
    task_types = ["recipe", "substitution", "expiry"]
    metrics = {}

    for strategy in strategies:
        strategy_rows = [r for r in rows if r["strategy"] == strategy]

        # Overall adherence
        adherence_vals = [int(r["constraint_adherence"]) for r in strategy_rows]
        overall_adherence = sum(adherence_vals) / len(adherence_vals) * 100

        # Per-task adherence
        task_adherence = {}
        for task in task_types:
            task_rows = [r for r in strategy_rows if r["task_type"] == task]
            if task_rows:
                vals = [int(r["constraint_adherence"]) for r in task_rows]
                task_adherence[task] = sum(vals) / len(vals) * 100
            else:
                task_adherence[task] = 0

        # Consistency
        consistency_vals = [float(r["consistency_score"]) for r in strategy_rows]
        mean_consistency = sum(consistency_vals) / len(consistency_vals)

        # Substitution validity
        sub_rows = [r for r in strategy_rows if r["substitution_validity"] != "N/A"]
        if sub_rows:
            validity = sum(int(r["substitution_validity"]) for r in sub_rows) / len(sub_rows) * 100
        else:
            validity = 0

        metrics[strategy] = {
            "overall_adherence": overall_adherence,
            "task_adherence": task_adherence,
            "consistency": mean_consistency,
            "substitution_validity": validity,
        }

    return metrics, strategies


def plot(metrics, strategies):
    labels = ["Zero-shot", "Few-shot", "Chain-of-thought"]
    x = np.arange(len(labels))
    width = 0.2

    # ── Figure 1: Adherence by task type ─────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    recipe = [metrics[s]["task_adherence"]["recipe"] for s in strategies]
    subst = [metrics[s]["task_adherence"]["substitution"] for s in strategies]
    expiry = [metrics[s]["task_adherence"]["expiry"] for s in strategies]
    overall = [metrics[s]["overall_adherence"] for s in strategies]

    ax1.bar(x - 1.5 * width, recipe, width, label="Recipe", color="#4C72B0")
    ax1.bar(x - 0.5 * width, subst, width, label="Substitution", color="#DD8452")
    ax1.bar(x + 0.5 * width, expiry, width, label="Expiry", color="#55A868")
    ax1.bar(x + 1.5 * width, overall, width, label="Overall", color="#C44E52", alpha=0.7)

    ax1.set_ylabel("Constraint Adherence (%)")
    ax1.set_title("Constraint Adherence by Task Type and Prompting Strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig("evaluation_adherence.png", dpi=150)
    print("Saved evaluation_adherence.png")

    # ── Figure 2: Consistency scores ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    consistency = [metrics[s]["consistency"] for s in strategies]
    bars = ax2.bar(labels, consistency, color=["#4C72B0", "#DD8452", "#55A868"], width=0.4)
    ax2.set_ylabel("Mean Consistency Score (Jaccard)")
    ax2.set_title("Consistency Score by Prompting Strategy")
    ax2.set_ylim(0, 0.6)
    for bar, val in zip(bars, consistency):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("evaluation_consistency.png", dpi=150)
    print("Saved evaluation_consistency.png")


def main():
    rows = load_results(INPUT_FILE)
    metrics, strategies = compute_metrics(rows)

    print("\nSummary:")
    for s in strategies:
        m = metrics[s]
        print(f"\n{s.upper()}")
        print(f"  Overall adherence : {m['overall_adherence']:.2f}%")
        print(f"  Recipe adherence  : {m['task_adherence']['recipe']:.2f}%")
        print(f"  Subst. adherence  : {m['task_adherence']['substitution']:.2f}%")
        print(f"  Expiry adherence  : {m['task_adherence']['expiry']:.2f}%")
        print(f"  Consistency       : {m['consistency']:.3f}")
        print(f"  Subst. validity   : {m['substitution_validity']:.2f}%")

    plot(metrics, strategies)


if __name__ == "__main__":
    main()