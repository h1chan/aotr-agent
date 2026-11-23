"""Ретайминг видео по реальным таймстампам кадров.

Если захват шёл медленнее номинального FPS (обычно из-за нагрузки на GPU/CPU),
video.mp4 проигрывается быстрее реального времени. Скрипт растягивает видео
до настоящей длительности по данным frames.jsonl.

Для обучения ретайминг не обязателен: датасет строится по frames.jsonl,
где лежат истинные таймстампы. Это нужно для просмотра человеком.

Использование: python3 retime.py ~/aotr-data/episode_0001
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def retime(episode: Path, crf: int = 23) -> None:
    video = episode / "video.mp4"
    frames_log = episode / "frames.jsonl"
    if not video.exists() or not frames_log.exists():
        raise SystemExit(f"нет video.mp4 или frames.jsonl в {episode}")

    ts = [json.loads(line)["t"] for line in frames_log.open()]
    if len(ts) < 2:
        raise SystemExit("слишком мало кадров")

    meta = json.loads((episode / "meta.json").read_text())
    nominal = meta["fps"]
    real_duration = ts[-1] - ts[0]
    encoded_duration = len(ts) / nominal
    ratio = real_duration / encoded_duration

    if abs(ratio - 1.0) < 0.02:
        print(f"{episode.name}: ретайминг не нужен (ratio={ratio:.3f})")
        return

    raw = episode / "video_raw.mp4"
    if not raw.exists():
        video.rename(raw)

    out = episode / "video.mp4"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(raw),
        "-filter:v", f"setpts=PTS*{ratio:.6f}",
        "-r", str(nominal), "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", str(crf), "-pix_fmt", "yuv420p", str(out),
    ]
    subprocess.run(cmd, check=True)
    print(f"{episode.name}: растянуто ×{ratio:.3f} "
          f"({encoded_duration:.1f} c → {real_duration:.1f} c)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        retime(Path(arg).expanduser())
