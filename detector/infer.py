"""Проверка детектора на кадрах: рисует найденных титанов.

  .venv/bin/python detector/infer.py FRAME.png [...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from train import Net, decode, resize


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Net()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def predict(model, ck, rgb: np.ndarray, threshold: float = 0.35):
    iw, ih = ck["in"]
    small = resize(rgb, iw, ih)
    x = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0) / 255
    with torch.no_grad():
        hm = torch.sigmoid(model(x))[0, 0].numpy()

    out_w, out_h = ck["out"]
    pts = []
    while True:
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        if hm[idx] < threshold:
            break
        y, x = idx
        pts.append((int(x / out_w * rgb.shape[1]), int(y / out_h * rgb.shape[0]),
                    float(hm[idx])))
        yy, xx = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
        hm = hm * (1 - np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 18))
    return pts, hm


def predict_components(model, ck, rgb: np.ndarray, threshold: float = 0.55,
                       min_area: int = 2):
    """Пики как центры связных пятен. Один титан даёт одну точку, а не пять."""
    from heuristic import components

    iw, ih = ck["in"]
    small = resize(rgb, iw, ih)
    x = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0) / 255
    with torch.no_grad():
        hm = torch.sigmoid(model(x))[0, 0].numpy()

    out_w, out_h = ck["out"]
    pts = []
    for x0, y0, x1, y1, _ in components(hm > threshold, min_area):
        patch = hm[y0:y1 + 1, x0:x1 + 1]
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        weight = patch.sum() + 1e-6
        cx = float((xx * patch).sum() / weight)
        cy = float((yy * patch).sum() / weight)
        pts.append((cx / out_w * rgb.shape[1], cy / out_h * rgb.shape[0],
                    float(patch.max())))
    pts.sort(key=lambda p: p[2], reverse=True)
    return pts, hm


def draw(rgb: np.ndarray, pts) -> np.ndarray:
    img = rgb.copy()
    for x, y, score in pts:
        r = 24
        img[max(0, y - r):y + r, max(0, x - 2):x + 2] = (0, 255, 0)
        img[max(0, y - 2):y + 2, max(0, x - r):x + r] = (0, 255, 0)
    return img


def save_png(path: Path, rgb: np.ndarray) -> None:
    ppm = path.with_suffix(".ppm")
    with open(ppm, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]))
        f.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ppm), str(path)],
                   check=True)
    ppm.unlink(missing_ok=True)


if __name__ == "__main__":
    model, ck = load_model(Path(__file__).parent / "model.pt")
    for arg in sys.argv[1:]:
        src = Path(arg)
        rgb = decode(src)
        pts, _ = predict(model, ck, rgb)
        print(f"{src.name}: {len(pts)} титанов", [round(p[2], 2) for p in pts])
        save_png(src.with_name(src.stem + "_pred.png"), draw(rgb, pts))
