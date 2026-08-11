"""Phase 3 -- per-image entropy (uncertainty) scoring.

Computes how confused the current model is about each pool image, using
softmax entropy over per-pixel class probabilities. High entropy = model
is unsure (probabilities spread across classes); low entropy = model is
confident (one class dominates). This score drives active learning's
image selection in select.py.

Wrapped in torch.no_grad() -- same discipline as Phase 2's evaluate.py --
since we are only measuring the model's current behavior, never training it.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.phase1_baseline.train import LitSegmentation
from src.phase2_gap.dataset import CropAndWeedSegmentation
from src.phase3_active_learning.split import get_pool_and_val_indices


@torch.no_grad()
def compute_pool_entropy(checkpoint_path: str, data_root: str = "data/cropandweed",
                          img_size: int = 256, batch_size: int = 4, device: str = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LitSegmentation.load_from_checkpoint(checkpoint_path)
    model = model.to(device).eval()

    full_dataset = CropAndWeedSegmentation(data_root, img_size=img_size)
    pool_idx, _ = get_pool_and_val_indices(len(full_dataset))
    pool_dataset = Subset(full_dataset, pool_idx.tolist())

    # shuffle=False is essential here -- we need to know exactly which
    # entropy score belongs to which original image index.
    loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    entropy_scores = {}
    running_idx = 0
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)

        probs = torch.softmax(logits, dim=1)
        pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)   # [batch, H, W]
        image_entropy = pixel_entropy.mean(dim=[1, 2])                  # [batch]

        for e in image_entropy.tolist():
            original_idx = int(pool_idx[running_idx])
            entropy_scores[original_idx] = e
            running_idx += 1

    return entropy_scores


def main():
    checkpoint_path = "outputs/checkpoints/unet/best.ckpt"
    scores = compute_pool_entropy(checkpoint_path)

    out_path = Path("outputs/phase3_pool_entropy.json")
    with open(out_path, "w") as f:
        json.dump(scores, f)

    values = list(scores.values())
    print(f"Computed entropy for {len(scores)} pool images.")
    print(f"Entropy range: min={min(values):.4f}, max={max(values):.4f}, mean={sum(values)/len(values):.4f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()