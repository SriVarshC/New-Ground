"""Phase 3 -- fixed pool/val split for CropAndWeed.

The val split is held out permanently: never eligible for active-learning
or random selection, never used for fine-tuning, in ANY experiment across
this phase. It exists solely to give every experiment (active-5/10/20,
random-5/10/20) a shared, fair, untouched basis for final comparison.

Uses a fixed random seed so this split is IDENTICAL every time it's
computed -- critical, since a shifting val set would make comparisons
across experiments meaningless.

Usage:
    from src.phase3_active_learning.split import get_pool_and_val_indices
    pool_idx, val_idx = get_pool_and_val_indices(total_len=7705)
"""
from __future__ import annotations

import numpy as np

SPLIT_SEED = 42          # fixed -- never change this once experiments have started
VAL_FRACTION = 0.20


def get_pool_and_val_indices(total_len: int, val_fraction: float = VAL_FRACTION,
                               seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pool_indices, val_indices), a fixed, reproducible partition
    of range(total_len). Same seed + same total_len ALWAYS produces the
    same split -- do not call this with a different seed mid-project."""
    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_len)
    rng.shuffle(all_indices)  # shuffled in-place, but deterministically given the seed

    val_size = int(total_len * val_fraction)
    val_indices = all_indices[:val_size]
    pool_indices = all_indices[val_size:]

    return pool_indices, val_indices


if __name__ == "__main__":
    from src.phase2_gap.dataset import CropAndWeedSegmentation

    ds = CropAndWeedSegmentation("data/cropandweed")
    pool_idx, val_idx = get_pool_and_val_indices(len(ds))

    print(f"Total images: {len(ds)}")
    print(f"Pool: {len(pool_idx)} ({len(pool_idx)/len(ds)*100:.1f}%)")
    print(f"Val:  {len(val_idx)} ({len(val_idx)/len(ds)*100:.1f}%)")

    # Reproducibility check: calling it again must give the exact same split.
    pool_idx2, val_idx2 = get_pool_and_val_indices(len(ds))
    assert np.array_equal(pool_idx, pool_idx2), "Split is not reproducible!"
    assert np.array_equal(val_idx, val_idx2), "Split is not reproducible!"
    print("Reproducibility check passed -- same split every call.")

    # No overlap check: pool and val must never share an image.
    overlap = set(pool_idx.tolist()) & set(val_idx.tolist())
    assert len(overlap) == 0, f"Pool and val overlap by {len(overlap)} images!"
    print("No-overlap check passed -- pool and val are fully disjoint.")