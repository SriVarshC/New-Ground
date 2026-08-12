"""Phase 2 -- measure the coverage-expansion gap.

Loads the untouched Phase 1 U-Net checkpoint and evaluates it on
CropAndWeed (the "new, unseen farm" stand-in), with zero fine-tuning.
Wrapped entirely in torch.no_grad() so the model's weights are
guaranteed unmodified -- the entire validity of this measurement
depends on the model being exactly what Phase 1 produced, nothing else.

Usage:
    python -m src.phase2_gap.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.common.metrics import iou_per_class
from src.phase1_baseline.train import LitSegmentation
from src.phase2_gap.dataset import NUM_CLASSES, CLASS_NAMES, CropAndWeedSegmentation


@torch.no_grad()
def evaluate_on_cropandweed(checkpoint_path: str, data_root: str = "data/cropandweed",
                             img_size: int = 256, batch_size: int = 4, device: str = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LitSegmentation.load_from_checkpoint(checkpoint_path)
    model = model.to(device).eval()

    dataset = CropAndWeedSegmentation(data_root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Accumulate intersection/union per class across the WHOLE dataset,
    # not averaged per-batch -- averaging per-batch and then averaging
    # those averages would over-weight small/rare batches unfairly.
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    correct_pixels = 0
    total_pixels = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

        for c in range(NUM_CLASSES):
            pred_c = preds == c
            target_c = masks == c
            total_intersection[c] += (pred_c & target_c).sum().item()
            total_union[c] += (pred_c | target_c).sum().item()

    iou_per_class_final = (total_intersection / total_union.clamp(min=1)).numpy()
    mean_iou = float(iou_per_class_final.mean())
    pixel_acc = correct_pixels / total_pixels

    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_acc,
        "iou_per_class": {name: float(iou_per_class_final[i]) for i, name in enumerate(CLASS_NAMES)},
        "num_images_evaluated": len(dataset),
    }


def main():
    checkpoint_path = "outputs/checkpoints/unet/best.ckpt"
    cropandweed_results = evaluate_on_cropandweed(checkpoint_path)

    print("CropAndWeed (unseen) results:")
    print(json.dumps(cropandweed_results, indent=2))

    # Load Phase 1's PhenoBench result for direct comparison.
    with open("outputs/phase1_results_unet.json") as f:
        phenobench_result = json.load(f)

    phenobench_iou = phenobench_result["best_val_mean_iou"]
    cropandweed_iou = cropandweed_results["mean_iou"]
    gap = phenobench_iou - cropandweed_iou

    report = {
        "phenobench_val_mean_iou": phenobench_iou,
        "cropandweed_mean_iou": cropandweed_iou,
        "coverage_expansion_gap": gap,
        "gap_percentage": (gap / phenobench_iou) * 100,
        "cropandweed_detail": cropandweed_results,
    }

    out_path = Path("outputs/phase2_gap_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved gap report -> {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()