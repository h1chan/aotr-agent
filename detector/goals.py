"""Считает вектор цели для каждого кадра датасета.

Для каждого кадра детектор находит лучшего титана и отдаёт:
  dx   смещение по горизонтали, -1 (слева) .. +1 (справа)
  dy   смещение по вертикали, -1 (сверху) .. +1 (снизу)
  conf уверенность
  size площадь пятна, грубый признак дистанции

  .venv/bin/python detector/goals.py ~/aotr-data/datasets/ds_e2
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from heuristic import components
from infer import load_model
from train import resize

HERE = Path(__file__).parent


def frames_from_video(episode: Path, limit: int, size=(200, 112)):
    meta = json.loads((episode / "meta.json").read_text())
    src_w, src_h = meta["size"]
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(episode / "video.mp4"),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
        stdout=subprocess.PIPE,
    )
    size = src_w * src_h * 3
    n = 0
    while n < limit:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
        yield resize(rgb, size[0], size[1])
        n += 1
    proc.stdout.close()
    proc.wait()


def goal_from_heatmap(hm: np.ndarray, threshold: float = 0.55):
    best = (0.0, 0.0, 0.0, 0.0)
    boxes = components(hm > threshold, 2)
    for x0, y0, x1, y1, area in boxes:
        patch = hm[y0:y1 + 1, x0:x1 + 1]
        if patch.max() < threshold:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        w = patch.sum() + 1e-6
        cx = float((xx * patch).sum() / w) / hm.shape[1]
        cy = float((yy * patch).sum() / w) / hm.shape[0]
        score = float(patch.max())
        if score > best[2]:
            best = (cx * 2 - 1, cy * 2 - 1, score, area / (hm.shape[0] * hm.shape[1]))
    return best


def main(dataset: Path) -> None:
    root = dataset.parent.parent  # ~/aotr-data
    episode = root / dataset.name.replace("ds_e", "episode_000")
    if not episode.exists():
        raise SystemExit(f"не нашёл исходный эпизод для {dataset} ({episode})")

    n = int(json.loads((dataset / "meta.json").read_text())["samples"])
    model, ck = load_model(HERE / "model4.pt")
    size = tuple(ck["in"])

    goals = np.zeros((n, 4), dtype=np.float32)
    batch: list[np.ndarray] = []
    batch_idx: list[int] = []
    i = 0

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            hms = torch.sigmoid(model(x)).numpy()[:, 0]
        for j, hm in zip(batch_idx, hms):
            goals[j] = goal_from_heatmap(hm)
        batch.clear()
        batch_idx.clear()

    for frame in frames_from_video(episode, n, size):
        batch.append(frame)
        batch_idx.append(i)
        i += 1
        if len(batch) >= 32:
            flush()
            if i % 2000 == 0:
                print(f"{i}/{n}", flush=True)
    flush()

    np.save(dataset / "goal.npy", goals)
    pos = int((goals[:, 2] > 0.55).sum())
    print(f"{dataset.name}: {i} кадров, с целью {pos} ({pos / max(1, i) * 100:.0f}%)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(Path(arg))
