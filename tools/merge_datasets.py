"""Склейка нескольких датасетов (папок с .npy) в один.

Нужно для DAgger: эпизодов много и они короткие, удобнее обучать на одном файле.

  python tools/merge_datasets.py OUT DIR1 DIR2 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def merge(out: Path, dirs: list[Path]) -> None:
    metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
    first = np.load(dirs[0] / "states.npy", mmap_mode="r")
    c, h, w = first.shape[1:]
    for d in dirs[1:]:
        s = np.load(d / "states.npy", mmap_mode="r")
        if s.shape[1:] != (c, h, w):
            raise SystemExit(f"{d}: форма {s.shape[1:]} не совпадает с {(c, h, w)}")

    counts = [len(np.load(d / "actions.npy")) for d in dirs]
    total = sum(counts)
    out.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        out / "states.npy", mode="w+", dtype=np.uint8, shape=(total, c, h, w)
    )

    actions = np.zeros((total, np.load(dirs[0] / "actions.npy").shape[1]), dtype=np.float32)
    times = np.zeros(total, dtype=np.float64)
    off = 0
    t_shift = 0.0
    for d, n in zip(dirs, counts):
        s = np.load(d / "states.npy", mmap_mode="r")
        states[off:off + n] = s[:n]
        actions[off:off + n] = np.load(d / "actions.npy")[:n]
        t = np.load(d / "times.npy")[:n]
        times[off:off + n] = t + t_shift
        t_shift = times[off + n - 1] + 1 / 15
        off += n
    states.flush()
    del states

    np.save(out / "actions.npy", actions)
    np.save(out / "times.npy", times)

    meta = dict(metas[0])
    meta.update({"samples": total, "merged_from": [d.name for d in dirs],
                 "merged_count": len(dirs)})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: {total} сэмплов из {len(dirs)} эпизодов")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    merge(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
