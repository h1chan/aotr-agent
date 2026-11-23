"""Метрики детектора: precision/recall по точкам на отложенных кадрах.

  .venv/bin/python detector/eval.py [model.pt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from infer import load_model, predict_components
from train import decode

HERE = Path(__file__).parent
DIST = 120  # пикселей, попадание


def main(model_path: Path, threshold: float = 0.3) -> None:
    model, ck = load_model(model_path)
    index = json.loads((HERE / "frames" / "index.json").read_text())
    labels = {}
    for line in (HERE / "labels.jsonl").open():
        line = line.strip()
        if line:
            row = json.loads(line)
            labels[row["i"]] = row["points"]

    ids = [it["i"] for it in index if it["i"] in labels]
    rng = np.random.RandomState(0).permutation(len(ids))
    val_ids = [ids[i] for i in rng[int(len(ids) * 0.8):]]

    tp = fp = fn = 0
    errors = []
    per_frame = []
    for i in val_ids:
        item = next(it for it in index if it["i"] == i)
        rgb = decode(HERE / "frames" / item["name"])
        pts, _ = predict_components(model, ck, rgb, threshold)
        gt = labels[i]
        used = set()
        frame_tp = 0
        for g in gt:
            best, best_j = 1e9, -1
            for j, (x, y, _) in enumerate(pts):
                if j in used:
                    continue
                d = ((x - g["x"]) ** 2 + (y - g["y"]) ** 2) ** 0.5
                if d < best:
                    best, best_j = d, j
            if best_j >= 0 and best <= DIST:
                used.add(best_j)
                tp += 1
                frame_tp += 1
                errors.append(best)
            else:
                fn += 1
        fp += len(pts) - len(used)
        per_frame.append((i, len(gt), frame_tp, len(pts)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(f"отложенных кадров {len(val_ids)}")
    print(f"TP {tp} FP {fp} FN {fn}")
    print(f"precision {precision:.3f} recall {recall:.3f} F1 {f1:.3f}")
    if errors:
        print(f"медианная ошибка попадания {np.median(errors):.0f} px")
    for i, ngt, hit, npred in per_frame[:15]:
        print(f"  кадр {i:3d}: разметка {ngt}, попало {hit}, предсказано {npred}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "model3.pt"
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    main(path, th)
