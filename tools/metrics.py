"""Читает экран результатов миссии и достаёт цифры.

Один OCR на прогон, а не на кадр: скрипт ищет среди снимков тот, где есть статистика,
и распознаёт только область с цифрами.

  .venv/bin/python tools/metrics.py ~/aotr-data/agent_disc2_run2
  .venv/bin/python tools/metrics.py shot1.png shot2.png --csv runs.csv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

# область панели статистики в долях кадра (1600x900)
CROP = (0.36, 0.17, 0.72, 0.53)


def crop_stats(img: Image.Image) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = (int(CROP[0] * w), int(CROP[1] * h),
                      int(CROP[2] * w), int(CROP[3] * h))
    part = img.crop((x0, y0, x1, y1)).convert("L")
    part = part.resize((part.width * 2, part.height * 2), Image.LANCZOS)
    return ImageOps.autocontrast(part)


def ocr(img: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        path = tmp.name
    try:
        out = subprocess.run(["tesseract", path, "stdout", "--psm", "6"],
                             capture_output=True, text=True, timeout=30).stdout
    finally:
        Path(path).unlink(missing_ok=True)
    return out


CONFUSIONS = str.maketrans({")": "0", "o": "0", "O": "0", "l": "1", "I": "1",
                            "S": "5", "B": "8", ",": "", ".": "", "|": "1"})


def parse(text: str) -> dict | None:
    flat = re.sub(r"\s+", " ", text.upper())

    def token(*labels: str) -> str | None:
        for lab in labels:
            m = re.search(lab + r"\s*([^\s]+)", flat)
            if m:
                return m.group(1)
        return None

    def number(*labels: str) -> int | None:
        raw = token(*labels)
        if raw is None:
            return None
        digits = re.sub(r"\D", "", raw.translate(CONFUSIONS))
        return int(digits) if digits else None

    time_raw = token(r"TIME\s*TAKEN")
    time_str = None
    if time_raw:
        bits = re.sub(r"[^\d:]", "", time_raw)
        if ":" in bits:
            time_str = bits

    damage = number(r"DAMAGE\s*DEALT")
    kills = number(r"TITANS\s*KILLED", "TITANSKILLED")
    crits = number(r"CRITICAL\s*HITS", "CRITICALHITS")

    if damage is None and kills is None:
        return None

    outcome = None
    if "MISSION FAILED" in flat:
        outcome = "failed"
    elif "MISSION COMPLETE" in flat or "MISSION SUCCESS" in flat:
        outcome = "complete"

    return {"outcome": outcome, "time": time_str,
            "damage": damage, "kills": kills, "crits": crits}


def read_image(path: Path) -> dict | None:
    try:
        img = Image.open(path)
    except OSError:
        return None
    return parse(ocr(crop_stats(img)))


def scan_dir(path: Path, check_last: int = 40) -> tuple[Path, dict] | None:
    shots = sorted(path.glob("*.png"))[-check_last:]
    for shot in reversed(shots):
        stats = read_image(shot)
        if stats:
            return shot, stats
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for path in args.paths:
        if path.is_dir():
            found = scan_dir(path)
            if not found:
                print(f"{path.name}: экран результатов не найден")
                continue
            shot, stats = found
            name = path.name
        else:
            stats = read_image(path)
            if not stats:
                print(f"{path.name}: не распозналось")
                continue
            name = path.stem
        row = {"run": name, **stats}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.csv and rows:
        import csv
        with args.csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if args.csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(rows)
        print(f"дописано в {args.csv}")


if __name__ == "__main__":
    main()
