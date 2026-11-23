"""Захват экрана через grim (Hyprland / wlr-screencopy).

Отдаёт кадры как numpy-массивы (H, W, 3) uint8, формат RGB.
grim умеет писать PPM (P6) в stdout — это самый быстрый путь без зависимостей.

Проверка: python3 capture.py --bench
"""

from __future__ import annotations

import subprocess
import time

import numpy as np


def parse_ppm(data: bytes) -> np.ndarray:
    """Разбор бинарного PPM (P6). Заголовок: P6\\n<w> <h>\\n<max>\\n."""
    if not data.startswith(b"P6"):
        raise ValueError("не PPM P6")
    fields: list[int] = []
    i = 2
    n = len(data)
    while len(fields) < 3:
        while i < n and data[i : i + 1].isspace():
            i += 1
        if i < n and data[i : i + 1] == b"#":
            while i < n and data[i : i + 1] != b"\n":
                i += 1
            continue
        start = i
        while i < n and not data[i : i + 1].isspace():
            i += 1
        fields.append(int(data[start:i]))
    i += 1  # один пробельный символ после maxval

    width, height, maxval = fields
    if maxval != 255:
        raise ValueError(f"ожидался maxval=255, получено {maxval}")

    buf = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=i)
    return buf.reshape(height, width, 3)


def grab(region: str | None = None, scale: float | None = None) -> np.ndarray:
    """Один кадр. region — 'x,y,w,h', scale — коэффициент уменьшения."""
    cmd = ["grim", "-t", "ppm"]
    if region:
        cmd += ["-g", region]
    if scale:
        cmd += ["-s", str(scale)]
    cmd.append("-")
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return parse_ppm(out)


def bench(n: int = 30, region: str | None = None, scale: float | None = None) -> None:
    # прогрев
    grab(region, scale)
    t0 = time.perf_counter()
    for _ in range(n):
        frame = grab(region, scale)
    dt = time.perf_counter() - t0
    fps = n / dt
    print(f"{n} кадров за {dt:.3f} c → {fps:.1f} FPS, {dt / n * 1000:.1f} мс/кадр")
    print(f"размер кадра: {frame.shape[1]}x{frame.shape[0]}, dtype={frame.dtype}")


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        region = None
        scale = None
        for i, arg in enumerate(sys.argv):
            if arg == "--region":
                region = sys.argv[i + 1]
            if arg == "--scale":
                scale = float(sys.argv[i + 1])
        bench(region=region, scale=scale)
    else:
        f = grab()
        print(f"кадр: {f.shape[1]}x{f.shape[0]}, средняя яркость {f.mean():.1f}")
