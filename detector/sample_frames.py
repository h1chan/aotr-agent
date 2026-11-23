"""Сбор кадров для разметки: ровно N штук, равномерно по всем записям.

  python3 detector/sample_frames.py [N]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "aotr-data"
OUT = Path(__file__).parent / "frames"


def main(n: int = 200) -> None:
    shots = sorted(p for p in ROOT.glob("**/full/*.png") if "_boxes" not in p.name)
    if not shots:
        raise SystemExit("нет кадров в ~/aotr-data/**/full")

    step = max(1, len(shots) / n)
    picked = [shots[int(i * step)] for i in range(min(n, len(shots)))]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    index = []
    for i, src in enumerate(picked):
        name = f"{i:04d}.png"
        shutil.copyfile(src, OUT / name)
        index.append({"i": i, "name": name, "src": str(src)})

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(picked)} кадров из {len(shots)} → {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
