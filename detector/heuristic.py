"""Поиск титанов по цвету кожи, без обучения.

Идея: у титанов ровный бежево-розовый цвет, крупные гладкие пятна. Если эвристика
работает, детектор не нужен: цель можно считать прямо в бою.

  python detector/heuristic.py FRAME.png [FRAME2.png ...]
Пишет рядом *_boxes.png с найденными рамками.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn + 1e-6

    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g - b) / diff) % 360)[mask]
    mask = mx == g
    h[mask] = (60 * ((b - r) / diff) + 120)[mask]
    mask = mx == b
    h[mask] = (60 * ((r - g) / diff) + 240)[mask]
    s = np.where(mx > 0, diff / (mx + 1e-6), 0)
    return h, s, mx


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    h, s, v = rgb_to_hsv(rgb)
    # бежевый и светло-розовый: тон 10..45 градусов, средняя насыщенность
    return (h >= 8) & (h <= 45) & (s >= 0.12) & (s <= 0.55) & (v >= 0.45)


def components(mask: np.ndarray, min_area: int):
    """Связные области через заливку. Возвращает список (x0, y0, x1, y1, area)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < x0: x0 = cx
                if cx > x1: x1 = cx
                if cy < y0: y0 = cy
                if cy > y1: y1 = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                boxes.append((x0, y0, x1, y1, area))
    return boxes


def find_titans(rgb: np.ndarray, scale: int = 8) -> list[tuple[int, int, int, int, int]]:
    """Крупные светлые пятна с низкой текстурой. Титаны ярче фона и гладкие."""
    h, w = rgb.shape[:2]
    small = rgb[::scale, ::scale]
    hh, ww = small.shape[:2]
    k = 4
    kh, kw = (hh // k) * k, (ww // k) * k
    _, s, v = rgb_to_hsv(small[:kh, :kw])
    block = v.reshape(kh // k, k, kw // k, k).mean((1, 3))
    blur = np.repeat(np.repeat(block, k, 0), k, 1)
    tex = np.abs(v - blur)

    mask = (v > 0.30) & (tex < 0.05)
    min_area = max(40, int(0.004 * mask.size))
    boxes = components(mask, min_area)

    out = []
    for x0, y0, x1, y1, area in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bh < 7 or bw < 4 or y0 <= 1:      # небо и всё, что липнет к верхнему краю
            continue
        aspect = bh / max(1, bw)
        if not (0.8 <= aspect <= 4.5):
            continue
        if area / max(1, bw * bh) < 0.5:
            continue
        patch_v = v[y0:y1, x0:x1]
        patch_s = s[y0:y1, x0:x1]
        if patch_v.mean() > 0.70 and patch_s.mean() < 0.12:  # облака
            continue
        out.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale, area))
    out.sort(key=lambda b: b[4], reverse=True)
    return out[:6]


def draw(rgb: np.ndarray, boxes) -> np.ndarray:
    img = rgb.copy()
    for x0, y0, x1, y1, _ in boxes:
        img[y0:y0 + 3, x0:x1] = (0, 255, 0)
        img[y1 - 3:y1, x0:x1] = (0, 255, 0)
        img[y0:y1, x0:x0 + 3] = (0, 255, 0)
        img[y0:y1, x1 - 3:x1] = (0, 255, 0)
    return img


def save_png(path: Path, rgb: np.ndarray) -> None:
    import subprocess
    ppm = path.with_suffix(".ppm")
    with open(ppm, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]))
        f.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ppm), str(path)],
                   check=True)
    ppm.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    import subprocess
    for arg in sys.argv[1:]:
        src = Path(arg)
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        # размер берём из ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, check=True).stdout.strip()
        w, h = (int(v) for v in probe.split(","))
        rgb = np.frombuffer(raw, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        boxes = find_titans(rgb)
        print(f"{src.name}: {len(boxes)} кандидатов")
        for b in boxes:
            print("   ", b[:4], "площадь", b[4])
        save_png(src.with_name(src.stem + "_boxes.png"), draw(rgb, boxes))
