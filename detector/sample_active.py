"""Активная выборка: 100 кадров, где текущий детектор не уверен.

Смысл: размечать не случайные кадры, а те, где модель ошибается или сомневается.
Такой кадр даёт больше пользы, чем ещё один лёгкий.

  .venv/bin/python detector/sample_active.py [N]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from infer import load_model, predict
from train import decode

ROOT = Path.home() / "aotr-data"
HERE = Path(__file__).parent
OUT = HERE / "frames"


def main(n: int = 100) -> None:
    model, ck = load_model(HERE / "model3.pt")
    index = json.loads((OUT / "index.json").read_text())
    used = {Path(e["src"]) for e in index}
    shots = [p for p in sorted(ROOT.glob("**/full/*.png"))
             if "_boxes" not in p.name and p not in used]

    scored = []
    for k, src in enumerate(shots):
        try:
            rgb = decode(src)
        except Exception:
            continue
        pts, _ = predict(model, ck, rgb, 0.35)
        best = max((p[2] for p in pts), default=0.0)
        scored.append((best, src))
        if k % 100 == 0:
            print(f"просканировано {k}/{len(shots)}", flush=True)

    uncertain = [s for s in scored if 0.45 <= s[0] <= 0.80]
    certain_neg = [s for s in scored if s[0] < 0.35]
    rng = np.random.RandomState(0)
    rng.shuffle(uncertain)
    rng.shuffle(certain_neg)

    picked = [s for _, s in uncertain[:int(n * 0.7)]]
    picked += [s for _, s in certain_neg[:n - len(picked)]]
    if len(picked) < n:  # добираем остальным
        rest = [s for _, s in scored if s not in picked]
        picked += rest[:n - len(picked)]

    base = len(index)
    for k, src in enumerate(picked):
        i = base + k
        name = f"{i:04d}.png"
        shutil.copyfile(src, OUT / name)
        index.append({"i": i, "name": name, "src": str(src)})

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"добавлено {len(picked)}, всего {len(index)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
