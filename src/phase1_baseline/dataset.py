"""PhenoBench dataset loader for the Phase 1 baseline.

Folder layout (confirmed against the real extracted dataset):
    data/phenobench/
        train/images/*.png
        train/semantics/*.png      # 16-bit pixel-wise class-id masks
        val/images/*.png
        val/semantics/*.png

Label remapping (design decision, see project notes):
    Raw PhenoBench semantic values: background=0, crop=1, weed=2,
    partial-crop=3, partial-weed=4.
    We merge partial-crop -> crop and partial-weed -> weed, since no
    downstream phase (gap measurement, active learning) distinguishes
    visibility -- only crop vs. weed vs. background matters to those.
    Final classes used everywhere in this project: background=0, crop=1, weed=2.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 3  # background, crop, weed (after merging partial classes)
CLASS_NAMES = ["background", "crop", "weed"]

# Raw label -> merged label. Anything not listed maps to itself.
_LABEL_REMAP = {
    0: 0,  # background
    1: 1,  # crop
    2: 2,  # weed
    3: 1,  # partial-crop -> crop
    4: 2,  # partial-weed -> weed
}


def _remap_mask(mask: np.ndarray) -> np.ndarray:
    """Apply the merge above to a raw semantic mask array."""
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for raw_val, merged_val in _LABEL_REMAP.items():
        remapped[mask == raw_val] = merged_val
    return remapped


class PhenoBenchSegmentation(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform=None, img_size: int = 256):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "semantics"
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"{self.image_dir} not found -- check that PhenoBench is extracted "
                f"under data/phenobench/ with train/val subfolders."
            )

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        # Images are standard 8-bit RGB.
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Masks are 16-bit PNGs -- IMREAD_UNCHANGED preserves the raw
        # label values. A plain IMREAD_GRAYSCALE would silently corrupt
        # these since it assumes 8-bit input.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        mask = _remap_mask(mask)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # INTER_NEAREST for masks specifically -- linear interpolation would
        # blend adjacent class IDs into meaningless in-between values (e.g.
        # averaging class 1 and class 2 into a bogus "1.5").

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/phenobench"
    ds = PhenoBenchSegmentation(root, split="train")
    print(f"Loaded {len(ds)} training images.")
    img, mask = ds[0]
    print("image shape:", img.shape, "mask shape:", mask.shape)
    print("mask unique values (should only be 0, 1, 2):", torch.unique(mask))