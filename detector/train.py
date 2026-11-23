"""Детектор титанов: предсказывает тепловую карту по кадру.

Вход 200x112 RGB, выход 50x28 (шаг 4). В месте титана гауссово пятно.
Модель крошечная, обучается на CPU за минуты.

  .venv/bin/python detector/train.py
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
FRAMES = HERE / "frames"
IN_W, IN_H = 200, 112
OUT_W, OUT_H = 50, 28
SIGMA = 2.0


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in probe.split(","))
    return np.frombuffer(raw, np.uint8, count=w * h * 3).reshape(h, w, 3)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    sh, sw = img.shape[:2]
    ys = (np.arange(h) * sh / h).astype(np.int32)
    xs = (np.arange(w) * sw / w).astype(np.int32)
    return img[ys][:, xs]


def heatmap(points: list[dict], src_w: int, src_h: int) -> np.ndarray:
    hm = np.zeros((OUT_H, OUT_W), dtype=np.float32)
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]
    for p in points:
        cx = p["x"] / src_w * OUT_W
        cy = p["y"] / src_h * OUT_H
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
        hm = np.maximum(hm, blob)
    return hm


def build() -> tuple[np.ndarray, np.ndarray, list[int]]:
    index = json.loads((FRAMES / "index.json").read_text())
    labels: dict[int, list[dict]] = {}
    for line in (HERE / "labels.jsonl").open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        labels[row["i"]] = row["points"]  # последняя разметка побеждает

    xs, ys, used = [], [], []
    for item in index:
        i = item["i"]
        if i not in labels:
            continue
        img = decode(FRAMES / item["name"])
        h, w = img.shape[:2]
        xs.append(resize(img, IN_W, IN_H))
        ys.append(heatmap(labels[i], w, h))
        used.append(i)
    return np.stack(xs), np.stack(ys)[:, None], used


def dice_loss(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(self.enc(x))
        return F.interpolate(h, size=(OUT_H, OUT_W), mode="bilinear",
                             align_corners=False)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=HERE / "model.pt")
    ap.add_argument("--dice", action="store_true", help="добавить Dice loss")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=112)
    args = ap.parse_args()
    global IN_W, IN_H, OUT_W, OUT_H
    IN_W, IN_H = args.width, args.height
    OUT_W, OUT_H = IN_W // 4, IN_H // 4
    torch.manual_seed(0)
    torch.set_num_threads(4)
    t0 = time.time()
    x, y, used = build()
    print(f"кадров {len(x)}, точек {int((y > 0.5).sum())}, сборка {time.time()-t0:.0f} с")

    n = len(x)
    perm = np.random.RandomState(0).permutation(n)
    cut = int(n * 0.8)
    tr, va = perm[:cut], perm[cut:]

    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    xt = torch.from_numpy(x[tr]).permute(0, 3, 1, 2).float().div_(255)
    yt = torch.from_numpy(y[tr])
    xv = torch.from_numpy(x[va]).permute(0, 3, 1, 2).float().div_(255)
    yv = torch.from_numpy(y[va])

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(xt))
        total = 0.0
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            xb = xt[idx].clone()
            yb = yt[idx]
            if np.random.rand() < 0.5:  # отражение кадра и разметки вместе
                xb = torch.flip(xb, dims=[3])
                yb = torch.flip(yb, dims=[3])
            xb = (xb * (0.8 + 0.4 * np.random.rand())).clamp(0, 1)
            logits = model(xb)
            loss = bce(logits, yb)
            if args.dice:
                loss = loss + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        loss_value = total / len(xt)

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val = bce(model(xv), yv).item()
            print(f"epoch {epoch:2d}: train {loss_value:.4f} val {val:.4f}")
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "in": [IN_W, IN_H],
                            "out": [OUT_W, OUT_H], "sigma": SIGMA,
                            "frames": used}, args.out)

    print(f"лучший val {best:.4f}, модель {args.out}")


if __name__ == "__main__":
    main()
