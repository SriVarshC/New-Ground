"""Real, timed inference latency -- not FLOPs, not parameter counts.

Reused after every phase's model changes (fine-tuning, compression) so
the accuracy-vs-latency tradeoff is tracked consistently project-wide.
"""
from __future__ import annotations

import time

import torch


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str = "cuda",
    n_warmup: int = 10,
    n_iters: int = 50,
) -> dict:
    """Returns mean/median latency in milliseconds per frame.

    GPU warm-up iterations are excluded, and torch.cuda.synchronize() is
    called around each timed iteration -- skipping this on GPU silently
    under-reports latency, since CUDA calls are asynchronous by default.
    """
    model = model.eval().to(device)
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(n_warmup):
        _ = model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iters):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    n = len(times_ms)
    return {
        "mean_ms": sum(times_ms) / n,
        "median_ms": times_ms[n // 2],
        "device": device,
        "input_shape": input_shape,
    }