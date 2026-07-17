"""Phase 1 training entrypoint.

VRAM-safe config for a 4GB card: small physical batch (2) + gradient
accumulation (8 steps -> effective batch of 16) + mixed precision (fp16).

Usage:
    python -m src.phase1_baseline.train --arch unet --epochs 1 --smoke-test
    python -m src.phase1_baseline.train --arch unet --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from src.common.latency import benchmark_latency
from src.common.metrics import iou_per_class
from src.phase1_baseline.dataset import NUM_CLASSES, PhenoBenchSegmentation
from src.phase1_baseline.model import SegmentationModule


class LitSegmentation(pl.LightningModule):
    """Thin Lightning wrapper: fills in only the problem-specific parts
    (forward pass, loss, metrics). Lightning itself handles gradient
    accumulation, mixed precision, GPU placement, and checkpointing."""

    def __init__(self, arch: str = "unet", lr: float = 1e-3, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = SegmentationModule(arch=arch, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        preds = logits.argmax(dim=1)
        acc = (preds == masks).float().mean()
        ious = iou_per_class(preds, masks, self.num_classes)
        mean_iou = float(torch.nan_to_num(torch.tensor(ious)).nanmean())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_mean_iou", mean_iou, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["unet", "deeplabv3plus", "segformer"], required=True)
    parser.add_argument("--data-root", default="data/phenobench")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Use a tiny 20-image subset to quickly verify the pipeline runs without OOM.")
    args = parser.parse_args()

    train_ds = PhenoBenchSegmentation(args.data_root, split="train", img_size=args.img_size)
    val_ds = PhenoBenchSegmentation(args.data_root, split="val", img_size=args.img_size)

    if args.smoke_test:
        train_ds = Subset(train_ds, range(20))
        val_ds = Subset(val_ds, range(10))
        print(f"[smoke test] using {len(train_ds)} train / {len(val_ds)} val images only")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = LitSegmentation(arch=args.arch, lr=args.lr)

    ckpt_dir = Path(args.out_dir) / "checkpoints" / args.arch
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best", monitor="val_mean_iou", mode="max", save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",              # mixed precision -- halves memory for most tensors
        accumulate_grad_batches=args.accumulate,  # VRAM trick: simulate batch_size*accumulate
        callbacks=[checkpoint_cb],
        log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, val_loader)

    if args.smoke_test:
        print("[smoke test] completed without OOM -- pipeline verified.")
        return

    # Real run: benchmark latency on the best checkpoint, save results.
    best_model = LitSegmentation.load_from_checkpoint(checkpoint_cb.best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latency_gpu = benchmark_latency(best_model.model, input_shape=(1, 3, args.img_size, args.img_size), device=device)
    latency_cpu = benchmark_latency(best_model.model, input_shape=(1, 3, args.img_size, args.img_size), device="cpu")

    results = {
        "arch": args.arch,
        "best_val_mean_iou": float(trainer.callback_metrics.get("val_mean_iou", -1)),
        "best_val_acc": float(trainer.callback_metrics.get("val_acc", -1)),
        "latency": {"gpu": latency_gpu, "cpu": latency_cpu},
        "checkpoint": str(checkpoint_cb.best_model_path),
    }
    results_path = Path(args.out_dir) / f"phase1_results_{args.arch}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()