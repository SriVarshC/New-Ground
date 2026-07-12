"""Phase 1 -- one consistent wrapper around three architecture families.

U-Net and DeepLabV3+ (from segmentation_models_pytorch) return a plain
tensor at full input resolution. SegFormer (from HuggingFace transformers)
returns a structured output object with a `.logits` tensor at roughly 1/4
input resolution -- both differences are handled inside forward() so
nothing outside this file needs to know which architecture is running.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from src.phase1_baseline.dataset import NUM_CLASSES


def build_backbone(arch: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Constructs the raw architecture only -- no wrapping logic here.
    Kept separate from SegmentationModule so 'which model to build' stays
    independent from 'how to run data through whatever was built'."""
    if arch == "unet":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=num_classes,
        )
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=num_classes,
        )
    if arch == "segformer":
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # pretrained head was for a different class count
        )
    raise ValueError(f"Unknown arch '{arch}'. Choose from: unet, deeplabv3plus, segformer")


class SegmentationModule(nn.Module):
    def __init__(self, arch: str = "unet", num_classes: int = NUM_CLASSES):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.model = build_backbone(arch, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw_output = self.model(image)

        if self.arch == "segformer":
            # Step 1: unwrap -- .logits holds the actual tensor
            tensor = raw_output.logits
            # Step 2: resize -- only possible now that we have a real tensor
            tensor = F.interpolate(
                tensor, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            return tensor

        # U-Net / DeepLabV3+ already return a plain tensor at full resolution
        return raw_output


if __name__ == "__main__":
    dummy_image = torch.randn(2, 3, 256, 256)  # batch of 2, matches dataset.py's output shape

    for arch in ["unet", "deeplabv3plus", "segformer"]:
        model = SegmentationModule(arch=arch)
        model.eval()
        with torch.no_grad():
            output = model(dummy_image)
        print(f"{arch:15s} -> output shape: {tuple(output.shape)}")