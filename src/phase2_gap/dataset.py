"""CropAndWeed dataset loader for Phase 2 -- the "new, unseen farm" stand-in.

Uses the CropsOrWeed9 pre-mapped variant (produced by cropandweed-dataset's
own map_dataset.py), which already merges CropAndWeed's 74 raw species down
to: 8 individual crop species (IDs 0-7) + 1 merged weed class (ID 8).
ID 9 was confirmed empirically (not documented in datasets.py) to be the
background/unlabeled value -- it appeared in every sampled mask file
regardless of which crop/weed species were present, the same "always
present" signature background classes show in PhenoBench.

We remap further, down to our project's standard 3-class scheme, matching
PhenoBenchSegmentation exactly so Phase 2's evaluation code can treat both
datasets identically: background=0, crop=1, weed=2.

Folder layout (confirmed via `dir` on the real extracted dataset):
    data/cropandweed/
        images/*.png
        labelIds/CropsOrWeed9/*.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 3  # background, crop, weed -- matches PhenoBenchSegmentation
CLASS_NAMES = ["background", "crop", "weed"]

# Raw CropsOrWeed9 label -> our 3-class scheme.
# 0-7: eight individual crop species, all merged to "crop".
# 8: already-merged "Weed" class in CropsOrWeed9.
# 9: background/unlabeled (empirically confirmed, not in datasets.py's own listing).
_LABEL_REMAP = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # crops -> crop
    8: 2,                                              # weed -> weed
    9: 0,                                              # background -> background
}


def _remap_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for raw_val, merged_val in _LABEL_REMAP.items():
        remapped[mask == raw_val] = merged_val
    return remapped


class CropAndWeedSegmentation(Dataset):
    def __init__(self, root: str | Path, transform=None, img_size: int = 256):
        self.root = Path(root)
        self.transform = transform
        self.img_size = img_size

        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "labelIds" / "CropsOrWeed9"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"{self.image_dir} not found -- check CropAndWeed extraction path.")

        # Only keep images that actually have a matching mask -- recall
        # CropsOrWeed9 masks are only saved if the image contains at least
        # one of the mapped species, so not every raw image has one.
        all_images = sorted(self.image_dir.glob("*.jpg"))
        self.image_paths = [p for p in all_images if (self.mask_dir / f"{p.stem}.png").exists()]
        if not self.image_paths:
            raise RuntimeError(f"No matching image/mask pairs found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        mask = _remap_mask(mask)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/cropandweed"
    ds = CropAndWeedSegmentation(root)
    print(f"Loaded {len(ds)} images with matching masks.")
    img, mask = ds[0]
    print("image shape:", img.shape, "mask shape:", mask.shape)
    print("mask unique values (should only be 0, 1, 2):", torch.unique(mask))