"""Сравнение предсказаний агента с тем, что делал человек.

Безопасный способ увидеть «первые шаги» агента: он не управляет игрой,
а проигрывает запись и показывает, что бы нажал.

Использование:
  .venv/bin/python eval_compare.py dataset.npz checkpoints/policy_v0.pt --start 8000 --frames 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train import N_KEYS, Policy

KEYS = ["w", "a", "s", "d", "q", "e", "space", "ctrl", "shift", "lmb", "rmb", "r"]


def fmt(vec: np.ndarray, th: float = 0.5) -> str:
    on = [k for k, v in zip(KEYS, vec) if v > th]
    return ",".join(on) if on else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--start", type=int, default=8000)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if args.dataset.is_dir():
        xs = np.load(args.dataset / "states.npy", mmap_mode="r")
        ys = np.load(args.dataset / "actions.npy")
    else:
        d = np.load(args.dataset)
        xs, ys = d["states"], d["actions"]

    lo = max(0, args.start)
    hi = min(len(xs), lo + args.frames)
    x = np.array(xs[lo:hi], dtype=np.float32) / 255.0
    y = ys[lo:hi]
    del xs, ys

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Policy(stack=x.shape[1])
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        logits, mouse = model(torch.from_numpy(x))
    probs = torch.sigmoid(logits).numpy()
    mdx = mouse[:, 0].numpy()

    print(f"кадры {lo}..{hi - 1}, порог {args.threshold}")
    print(f"{'#':>6}  {'человек':<28} {'агент':<28} мышь")
    for i in range(hi - lo):
        t = fmt(y[i, :N_KEYS])
        p = fmt(probs[i], args.threshold)
        mark = "" if t == p else "  ←"
        print(f"{lo + i:6d}  {t:<28} {p:<28} {mdx[i] * 50:+6.1f}{mark}")

    pred = (probs > args.threshold).astype(np.float32)
    true = y[:, :N_KEYS]
    print("\nточность по клавишам (доля кадров, где предсказание совпало):")
    for k, name in enumerate(KEYS):
        acc = float((pred[:, k] == true[:, k]).mean())
        pos = float(true[:, k].mean())
        print(f"  {name:6s} acc {acc:.3f}  (человек держал {pos:.2f})")
    print(f"\nсредняя точность: {float((pred == true).mean()):.3f}")
    corr = np.corrcoef(mdx, y[:, N_KEYS])[0, 1] if np.std(mdx) > 1e-6 else float("nan")
    print(f"корреляция мыши: {corr:+.3f}" if corr == corr else "корреляция мыши: н/д")


if __name__ == "__main__":
    main()
