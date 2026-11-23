"""Детектор на предобученном MobileNetV3-small.

Почему он: сеть уже видела миллионы фотографий и знает, как выглядят текстуры,
силуэты и объекты. На 300 размеченных кадрах это компенсирует нехватку данных
лучше, чем увеличение разрешения.

  .venv/bin/python detector/train_mb.py --width 320 --height 180
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import train as T

HERE = Path(__file__).parent
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MobileNetDetector(nn.Module):
    def __init__(self, out_h: int, out_w: int):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.stem = backbone.features[:9]     # stride ~16, 48 каналов
        self.deep = backbone.features[9:]     # stride 32, 576 каналов
        self.head = nn.Sequential(
            nn.Conv2d(48 + 576, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.out_h, self.out_w = out_h, out_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - MEAN) / STD
        fine = self.stem(x)
        coarse = self.deep(fine)
        coarse = F.interpolate(coarse, size=fine.shape[2:], mode="bilinear",
                               align_corners=False)
        h = self.head(torch.cat([fine, coarse], dim=1))
        return F.interpolate(h, size=(self.out_h, self.out_w), mode="bilinear",
                             align_corners=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=HERE / "model_mb.pt")
    args = ap.parse_args()

    T.IN_W, T.IN_H = args.width, args.height
    T.OUT_W, T.OUT_H = args.width // 4, args.height // 4
    torch.manual_seed(0)
    torch.set_num_threads(4)

    t0 = time.time()
    x, y, used = T.build()
    print(f"кадров {len(x)}, сборка {time.time()-t0:.0f} с", flush=True)

    n = len(x)
    perm = np.random.RandomState(0).permutation(n)
    cut = int(n * 0.8)
    tr, va = perm[:cut], perm[cut:]

    model = MobileNetDetector(T.OUT_H, T.OUT_W)
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
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = xt[idx].clone()
            yb = yt[idx]
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
                yb = torch.flip(yb, dims=[3])
            xb = (xb * (0.8 + 0.4 * np.random.rand())).clamp(0, 1)
            loss = bce(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val = bce(model(xv), yv).item()
            print(f"epoch {epoch:2d}: train {total/len(xt):.4f} val {val:.4f}",
                  flush=True)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "in": [T.IN_W, T.IN_H],
                            "out": [T.OUT_W, T.OUT_H], "sigma": T.SIGMA,
                            "backbone": "mobilenet_v3_small", "frames": used},
                           args.out)

    print(f"лучший val {best:.4f} → {args.out}")


if __name__ == "__main__":
    main()
