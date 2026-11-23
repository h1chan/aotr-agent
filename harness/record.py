"""Запись демонстраций: экран + реальный ввод игрока.

Пишет видео (h264 через ffmpeg) и события клавиатуры/мыши с таймстампами.
Формат эпизода:

    episode_0001/
      video.mp4        # 400x225, 15 fps, h264
      input.jsonl      # {"t": <monotonic>, "type": ..., "code": ..., "value": ...}
      frames.jsonl     # {"t": <monotonic>, "idx": N}
      meta.json        # дата, разрешение, fps, устройства

Управление: F9 — старт/стоп записи, F10 — выход.

Требования: пользователь в группе `input` (чтение /dev/input/event*),
grim, ffmpeg.

Запуск: python3 record.py --out ~/aotr-data
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import capture

EVENT_FMT = "<qqHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)  # 24

EV_SYN, EV_KEY, EV_REL = 0, 1, 2

# Roblox перехватывает F9 (консоль) и F10 (графика), поэтому по умолчанию
# используем клавиши, которые игра не трогает. Можно переопределить через CLI.
HOTKEYS = {
    "scrolllock": 70,
    "pause": 119,
    "f7": 65,
    "f8": 66,
    "grave": 41,      # ` / ~
    "insert": 110,
    "home": 102,
    "end": 107,
    "pagedown": 109,
    "delete": 111,
}

# EVIOCGBIT(ev, len) = _IOC(_IOC_READ, 'E', 0x20 + ev, len)
def _eviocgbit(ev: int, length: int) -> int:
    return (2 << 30) | (length << 16) | (ord("E") << 8) | (0x20 + ev)


def _read_bits(fd: int, ev: int, length: int) -> bytes:
    buf = bytearray(length)
    fcntl.ioctl(fd, _eviocgbit(ev, length), buf, True)
    return bytes(buf)


def _bit_set(bits: bytes, code: int) -> bool:
    return code // 8 < len(bits) and bool(bits[code // 8] & (1 << (code % 8)))


def discover_devices() -> tuple[list[str], list[str]]:
    """Возвращает (клавиатуры, мыши) как пути /dev/input/eventN."""
    keyboards: list[str] = []
    mice: list[str] = []
    for entry in sorted(Path("/dev/input").glob("event*")):
        try:
            fd = os.open(entry, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            continue
        try:
            keys = _read_bits(fd, EV_KEY, 96)
            rels = _read_bits(fd, EV_REL, 8)
        finally:
            os.close(fd)
        has_mouse = _bit_set(rels, 0) and _bit_set(rels, 1)  # REL_X, REL_Y
        has_keyboard = _bit_set(keys, 30) and _bit_set(keys, 31)  # KEY_A, KEY_S
        if has_mouse:
            mice.append(str(entry))
        if has_keyboard and not has_mouse:
            keyboards.append(str(entry))
    return keyboards, mice


def device_name(path: str) -> str:
    try:
        with open(f"/sys/class/input/{Path(path).name}/device/name") as f:
            return f.read().strip()
    except OSError:
        return "?"


class Recorder:
    def __init__(self, out: Path, fps: int = 15, width: int = 800, height: int = 450,
                 snapshot_every: float = 5.0, tag: str = "",
                 toggle_key: int = 65, quit_key: int = 66):
        self.toggle_key = toggle_key
        self.quit_key = quit_key
        self.tag = tag
        self.out = out
        self.fps = fps
        self.width = width
        self.height = height
        self.snapshot_every = snapshot_every
        self.last_snapshot = 0.0
        self.recording = False
        self.episode = 0
        self.dir: Path | None = None
        self.ffmpeg: subprocess.Popen | None = None
        self.input_log = None
        self.frames_log = None
        self.frame_idx = 0
        self.t0 = 0.0

    # --- эпизод ---

    def _next_episode(self) -> int:
        """Следующий свободный номер: новый запуск не должен затирать старые записи."""
        used = [int(p.name.split("_")[1]) for p in self.out.glob("episode_*")
                if p.name.split("_")[1].isdigit()]
        return max(used, default=0) + 1

    def start(self) -> None:
        # h264 + yuv420p требуют чётных сторон — приводим до запуска ffmpeg
        self.width -= self.width % 2
        self.height -= self.height % 2
        self.episode = self._next_episode()
        self.dir = self.out / f"episode_{self.episode:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "full").mkdir(exist_ok=True)
        self.last_snapshot = 0.0
        self.input_log = open(self.dir / "input.jsonl", "w")
        self.frames_log = open(self.dir / "frames.jsonl", "w")
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(self.dir / "video.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        self.frame_idx = 0
        self.t0 = time.monotonic()
        self.recording = True
        print(f"[rec] старт {self.dir}")

    def stop(self) -> None:
        self.recording = False
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.ffmpeg = None
        for f in (self.input_log, self.frames_log):
            if f:
                f.close()
        self.input_log = self.frames_log = None
        if self.dir:
            duration = max(1e-6, time.monotonic() - self.t0)
            measured = self.frame_idx / duration
            meta = {
                "episode": self.episode,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fps": self.fps,
                "measured_fps": round(measured, 2),
                "size": [self.width, self.height],
                "frames": self.frame_idx,
                "duration_s": round(duration, 2),
                "tag": self.tag,
            }
            if abs(measured - self.fps) / self.fps > 0.1:
                print(f"[rec] ВНИМАНИЕ: реальный FPS {measured:.1f}, а видео помечено как "
                      f"{self.fps}. Тайминги кадров сохранены в frames.jsonl, "
                      f"видео нужно ретаймить перед просмотром.")
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[rec] стоп {self.dir}: {self.frame_idx} кадров, {meta['duration_s']} c")

    # --- ввод ---

    def log_event(self, path: str, t: float, etype: int, code: int, value: int) -> None:
        if not self.recording or not self.input_log:
            return
        # SYN и MSC — служебный шум, для обучения не нужны
        if etype in (EV_SYN, 4):
            return
        self.input_log.write(
            json.dumps({"t": round(t, 6), "dev": Path(path).name,
                        "type": etype, "code": code, "value": value}) + "\n"
        )

    # --- кадры ---

    def grab_frame(self) -> None:
        if not self.recording or not self.ffmpeg:
            return
        frame = capture.grab()
        h, w = frame.shape[:2]
        sy, sx = max(1, h // self.height), max(1, w // self.width)
        small = frame[: self.height * sy : sy, : self.width * sx : sx]
        # страховка: кадр обязан совпадать с параметрами ffmpeg
        if small.shape[0] < self.height or small.shape[1] < self.width:
            padded = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            padded[: small.shape[0], : small.shape[1]] = small
            small = padded
        elif small.shape[0] != self.height or small.shape[1] != self.width:
            small = small[: self.height, : self.width]
        try:
            self.ffmpeg.stdin.write(small.tobytes())
        except BrokenPipeError:
            print("[rec] ffmpeg упал, запись остановлена")
            self.stop()
            return
        self.frames_log.write(
            json.dumps({"t": round(time.monotonic(), 6), "idx": self.frame_idx}) + "\n"
        )

        # Полноразмерный кадр раз в N секунд — для чтения HUD и ручной проверки.
        # Запускаем асинхронно: синхронный grim с PNG блокировал цикл на ~1 секунду.
        now = time.monotonic()
        if now - self.last_snapshot >= self.snapshot_every:
            snapshot = self.dir / "full" / f"{self.frame_idx:06d}.png"
            subprocess.Popen(["grim", "-t", "png", str(snapshot)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.last_snapshot = now

        self.frame_idx += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "aotr-data")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--snapshot-every", type=float, default=5.0,
                    help="раз в сколько секунд сохранять полноразмерный кадр")
    ap.add_argument("--tag", type=str, default="",
                    help="метка эпизода: аккаунт/грейд/семья/сложность, например 'main-E+-normal'")
    ap.add_argument("--toggle-key", default="f7", choices=sorted(HOTKEYS),
                    help="клавиша старт/стоп записи (по умолчанию f7)")
    ap.add_argument("--quit-key", default="f8", choices=sorted(HOTKEYS),
                    help="клавиша выхода (по умолчанию f8)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    keyboards, mice = discover_devices()
    if not keyboards:
        print("Нет доступа к /dev/input/event*. Выполни:\n"
              "  sudo usermod -aG input $USER\nи перелогинься.", file=sys.stderr)
        return 1

    print("Клавиатуры:", ", ".join(f"{p} ({device_name(p)})" for p in keyboards))
    print("Мыши:", ", ".join(f"{p} ({device_name(p)})" for p in mice) or "нет")

    fds = {}
    for path in keyboards + mice:
        fds[os.open(path, os.O_RDONLY | os.O_NONBLOCK)] = path

    rec = Recorder(
        args.out, args.fps, args.width, args.height, args.snapshot_every, args.tag,
        toggle_key=HOTKEYS[args.toggle_key], quit_key=HOTKEYS[args.quit_key],
    )
    print(f"{args.toggle_key.upper()} — старт/стоп записи, "
          f"{args.quit_key.upper()} — выход.")

    next_frame = 0.0
    try:
        while True:
            timeout = max(0.0, next_frame - time.monotonic()) if rec.recording else 0.05
            ready, _, _ = select.select(list(fds), [], [], timeout)

            for fd in ready:
                path = fds[fd]
                try:
                    data = os.read(fd, EVENT_SIZE * 64)
                except BlockingIOError:
                    continue
                for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                    _, _, etype, code, value = struct.unpack_from(EVENT_FMT, data, off)
                    t = time.monotonic()
                    # служебные клавиши не пишем в датасет вообще (ни down, ни up)
                    if etype == EV_KEY and code == rec.toggle_key:
                        if value == 1:
                            rec.stop() if rec.recording else rec.start()
                        continue
                    if etype == EV_KEY and code == rec.quit_key:
                        if value == 1:
                            raise KeyboardInterrupt
                        continue
                    rec.log_event(path, t, etype, code, value)

            now = time.monotonic()
            if rec.recording and now >= next_frame:
                rec.grab_frame()
                # абсолютное расписание: захват не «съедает» интервал
                next_frame += 1.0 / rec.fps
                if next_frame < now:          # не успеваем — не догоняем пачкой
                    next_frame = now + 1.0 / rec.fps
    except KeyboardInterrupt:
        pass
    finally:
        if rec.recording:
            rec.stop()
        for fd in fds:
            os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
