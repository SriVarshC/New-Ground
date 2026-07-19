"""Aggregates outputs/phase1_results_*.json into the Phase 1 deliverable:
one comparison table (CSV) + one accuracy-vs-latency chart (PNG).

Usage:
    python -m src.phase1_baseline.compare
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    out_dir = Path("outputs")
    rows = []
    for path in sorted(glob.glob(str(out_dir / "phase1_results_*.json"))):
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "arch": r["arch"],
            "val_mean_iou": r["best_val_mean_iou"],
            "val_acc": r["best_val_acc"],
            "latency_gpu_ms": r["latency"]["gpu"]["mean_ms"],
            "latency_cpu_ms": r["latency"]["cpu"]["mean_ms"],
        })

    if not rows:
        raise SystemExit("No phase1_results_*.json found in outputs/ -- run train.py for each architecture first.")

    df = pd.DataFrame(rows).sort_values("val_mean_iou", ascending=False)
    print(df.to_string(index=False))

    table_path = out_dir / "phase1_comparison_table.csv"
    df.to_csv(table_path, index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["latency_gpu_ms"], df["val_mean_iou"], s=100)
    for _, row in df.iterrows():
        ax.annotate(row["arch"], (row["latency_gpu_ms"], row["val_mean_iou"]),
                     textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("GPU inference latency (ms/frame)")
    ax.set_ylabel("Validation mean IoU")
    ax.set_title("Phase 1: Accuracy vs. Latency Tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    chart_path = fig_dir / "phase1_accuracy_vs_latency.png"
    fig.savefig(chart_path, dpi=150)

    print(f"\nSaved table -> {table_path}")
    print(f"Saved chart -> {chart_path}")


if __name__ == "__main__":
    main()