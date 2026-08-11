"""Phase 3 -- renders the recovery curve: % of pool labeled vs. val mean
IoU, for both active and random selection, against the 0% baseline and
the original PhenoBench in-domain performance (the practical ceiling).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt


def main():
    with open("outputs/phase3_finetune_results.json") as f:
        results = json.load(f)

    with open("outputs/phase1_results_unet.json") as f:
        phenobench_iou = json.load(f)["best_val_mean_iou"]

    budgets = [0, 5, 10, 20]
    active = [results["0pct_baseline"]] + [results[f"active_{b}pct"] for b in [5, 10, 20]]
    random_ = [results["0pct_baseline"]] + [results[f"random_{b}pct"] for b in [5, 10, 20]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(budgets, active, marker="o", linewidth=2, label="Active (entropy-based)")
    ax.plot(budgets, random_, marker="s", linewidth=2, label="Random sampling")
    ax.axhline(phenobench_iou, color="gray", linestyle="--", linewidth=1.5,
               label=f"PhenoBench in-domain ceiling ({phenobench_iou:.3f})")

    ax.set_xlabel("% of pool labeled")
    ax.set_ylabel("Val Mean IoU (on CropAndWeed)")
    ax.set_title("Phase 3: Gap Recovery -- Active Learning vs. Random Sampling")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = "outputs/figures/phase3_recovery_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")

    print("\nSummary:")
    print(f"  0% baseline:  {results['0pct_baseline']:.4f}")
    for b in [5, 10, 20]:
        gap_recovered = (results[f'active_{b}pct'] - results['0pct_baseline']) / (phenobench_iou - results['0pct_baseline']) * 100
        gap_recovered_r = (results[f'random_{b}pct'] - results['0pct_baseline']) / (phenobench_iou - results['0pct_baseline']) * 100
        print(f"  {b}% budget -> active: {results[f'active_{b}pct']:.4f} ({gap_recovered:.1f}% of gap recovered), "
              f"random: {results[f'random_{b}pct']:.4f} ({gap_recovered_r:.1f}% of gap recovered)")


if __name__ == "__main__":
    main()