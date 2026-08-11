"""Phase 3 -- select which pool images get "labeled" (revealed) at a given
budget, using two strategies to compare:
  - active: the N highest-entropy (most uncertain) images
  - random: N images chosen uniformly at random (fixed seed, reproducible)

Both operate ONLY on the pool (never the held-out val set) and both are
given the exact same budget N at each comparison point, so any difference
in downstream fine-tuned accuracy is attributable to selection strategy,
not to differing amounts of data.
"""
from __future__ import annotations

import json

import numpy as np

RANDOM_SELECT_SEED = 123   # different from split.py's seed on purpose --
                            # these are two independent random decisions


def select_active(entropy_scores: dict, budget_fraction: float) -> list[int]:
    """Returns the N highest-entropy pool image indices, N = budget_fraction * pool size."""
    n = int(len(entropy_scores) * budget_fraction)
    # Sort by entropy descending -- most uncertain first.
    ranked = sorted(entropy_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [int(idx) for idx, _ in ranked[:n]]
    return selected


def select_random(entropy_scores: dict, budget_fraction: float, seed: int = RANDOM_SELECT_SEED) -> list[int]:
    """Returns N randomly chosen pool image indices, same N as select_active
    would choose for the same budget -- fixed seed for reproducibility."""
    n = int(len(entropy_scores) * budget_fraction)
    rng = np.random.default_rng(seed)
    all_pool_indices = np.array(list(entropy_scores.keys()))
    chosen = rng.choice(all_pool_indices, size=n, replace=False)
    return [int(idx) for idx in chosen]


def main():
    with open("outputs/phase3_pool_entropy.json") as f:
        entropy_scores = {int(k): v for k, v in json.load(f).items()}

    budgets = [0.05, 0.10, 0.20]
    selections = {}

    for budget in budgets:
        active_idx = select_active(entropy_scores, budget)
        random_idx = select_random(entropy_scores, budget)

        assert len(active_idx) == len(random_idx), "Active and random selections must be the same size!"

        selections[f"active_{int(budget*100)}pct"] = active_idx
        selections[f"random_{int(budget*100)}pct"] = random_idx

        print(f"Budget {int(budget*100)}%: {len(active_idx)} images selected "
              f"(active mean entropy: {np.mean([entropy_scores[i] for i in active_idx]):.4f}, "
              f"random mean entropy: {np.mean([entropy_scores[i] for i in random_idx]):.4f})")

    with open("outputs/phase3_selections.json", "w") as f:
        json.dump(selections, f)
    print("Saved -> outputs/phase3_selections.json")


if __name__ == "__main__":
    main()