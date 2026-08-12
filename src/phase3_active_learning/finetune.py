"""Phase 3 -- fine-tune U-Net's Phase 1 checkpoint on each active/random
selection at each budget, then evaluate on the FIXED held-out val set.

Each run starts fresh from the original Phase 1 checkpoint (never chained
fine-tunes on top of a previous fine-tune), uses a lower learning rate and
few epochs to avoid catastrophic forgetting of the original PhenoBench
training, and is evaluated against the identical val split every time so
the 6 results are directly comparable.
"""
from __future__ import annotations

import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from src.common.metrics import iou_per_class
from src.phase1_baseline.train import LitSegmentation
from src.phase2_gap.dataset import NUM_CLASSES, CropAndWeedSegmentation
from src.phase3_active_learning.split import get_pool_and_val_indices

FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-4          # lower than Phase 1's 1e-3 -- gentle nudge, not a fresh train
BATCH_SIZE = 2
ACCUMULATE = 8


@torch.no_grad()
def evaluate_on_val(model, val_loader, device) -> float:
    """Same accumulation-based IoU logic as Phase 2's evaluate.py."""
    model.eval()
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images).argmax(dim=1)
        for c in range(NUM_CLASSES):
            pred_c, target_c = preds == c, masks == c
            total_intersection[c] += (pred_c & target_c).sum().item()
            total_union[c] += (pred_c | target_c).sum().item()

    ious = (total_intersection / total_union.clamp(min=1)).numpy()
    return float(ious.mean())


def finetune_one_run(name: str, selected_indices: list[int], val_indices,
                      base_checkpoint: str, data_root: str = "data/cropandweed") -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fresh load every time -- never fine-tune on top of a previous fine-tune.
    model = LitSegmentation.load_from_checkpoint(base_checkpoint, lr=FINETUNE_LR)

    full_dataset = CropAndWeedSegmentation(data_root)
    train_subset = Subset(full_dataset, selected_indices)
    val_subset = Subset(full_dataset, val_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    trainer = pl.Trainer(
        max_epochs=FINETUNE_EPOCHS, accelerator="auto", devices=1,
        precision="16-mixed", accumulate_grad_batches=ACCUMULATE,
        enable_checkpointing=False, logger=False,
    )
    trainer.fit(model, train_loader)

    model = model.to(device)
    final_iou = evaluate_on_val(model, val_loader, device)
    print(f"[{name}] {len(selected_indices)} images, {FINETUNE_EPOCHS} epochs -> val mean IoU: {final_iou:.4f}")
    return final_iou


def main():
    base_checkpoint = "outputs/checkpoints/unet/best.ckpt"

    with open("outputs/phase3_selections.json") as f:
        selections = json.load(f)

    full_dataset = CropAndWeedSegmentation("data/cropandweed")
    _, val_indices = get_pool_and_val_indices(len(full_dataset))

    # Baseline: the untouched Phase 1 checkpoint's performance on this same
    # val set, before ANY fine-tuning -- this is Phase 2's number, our
    # zero-label starting point for the recovery curve.
    results = {}
    zero_shot_model = LitSegmentation.load_from_checkpoint(base_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zero_shot_model = zero_shot_model.to(device)
    val_subset = Subset(full_dataset, val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    results["0pct_baseline"] = evaluate_on_val(zero_shot_model, val_loader, device)
    print(f"[0pct_baseline] val mean IoU: {results['0pct_baseline']:.4f}")

    for name, indices in selections.items():
        results[name] = finetune_one_run(name, indices, val_indices, base_checkpoint)

    with open("outputs/phase3_finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> outputs/phase3_finetune_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()