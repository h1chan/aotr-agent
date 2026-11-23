"""Сборка датасета из эпизода: кадры -> состояния, ввод -> действия.

Выравнивание по frames.jsonl (истинные таймстампы), а не по PTS видео.
Для кадра i действие — это то, что игрок делал в интервале [t_i, t_{i+1}).

Выход: dataset.npz
  states  uint8  (N, stack, H, W)   — стек серых кадров
  actions float32 (N, 12)           — 10 клавиш/кнопок + dx, dy
  times   float64 (N,)              — таймстамп кадра, сек от начала
  meta    json-строка

Использование:
  python3 dataset.py ~/aotr-data/episode_0001
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# linux input codes
KEY_W, KEY_A, KEY_S, KEY_D = 17, 30, 31, 32
KEY_Q, KEY_E, KEY_R = 16, 18, 19
KEY_SPACE, KEY_LEFTCTRL, KEY_LEFTSHIFT = 57, 29, 42
BTN_LEFT, BTN_RIGHT = 0x110, 0x111
REL_X, REL_Y = 0x00, 0x01

# порядок колонок в actions: сначала клавиши, потом два столбца мыши (dx, dy)
ACTION_NAMES = ["w", "a", "s", "d", "q", "e", "space", "ctrl", "shift", "lmb", "rmb", "r"]
N_KEYS = len(ACTION_NAMES)
N_ACTIONS = N_KEYS + 2

_KEYMAP = {
    KEY_W: "w", KEY_A: "a", KEY_S: "s", KEY_D: "d",
    KEY_Q: "q", KEY_E: "e", KEY_SPACE: "space",
    KEY_LEFTCTRL: "ctrl", KEY_LEFTSHIFT: "shift",
    KEY_R: "r", BTN_LEFT: "lmb", BTN_RIGHT: "rmb",
}


def load_times(episode: Path) -> np.ndarray:
    return np.array([json.loads(line)["t"] for line in (episode / "frames.jsonl").open()],
                    dtype=np.float64)


def load_events(episode: Path) -> list[dict]:
    return [json.loads(line) for line in (episode / "input.jsonl").open()]


MOUSE_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
MOUSE_MAX = 40.0


def mouse_to_discrete(dx: float, dy: float, deadzone: float = 4.0) -> tuple[int, float]:
    """9 классов: 8 направлений и «не двигать». Плюс величина 0..1."""
    length = (dx * dx + dy * dy) ** 0.5
    if length < deadzone:
        return 8, 0.0
    angle = np.degrees(np.arctan2(dy, dx))  # 0 = вправо, 90 = вниз
    idx = int(((angle + 22.5) % 360) // 45)
    return idx, float(min(1.0, length / MOUSE_MAX))


def build_actions(times: np.ndarray, events: list[dict],
                  mouse_mode: str = "delta") -> np.ndarray:
    """(N, 12): состояние клавиш на момент кадра + мышь за интервал.

    mouse_mode="delta"    — две дельты, как раньше;
    mouse_mode="discrete" — класс направления (0..8) и величина (0..1).
    """
    n = len(times)
    actions = np.zeros((n, N_ACTIONS), dtype=np.float32)
    held = {name: 0.0 for name in ACTION_NAMES}

    j = 0
    for i in range(n):
        t0 = times[i]
        t1 = times[i + 1] if i + 1 < n else t0 + 1.0 / 15.0

        dx = dy = 0.0
        while j < len(events) and events[j]["t"] < t1:
            e = events[j]
            t = e["t"]
            # события до первого кадра применяем сразу (начальное состояние)
            if e["type"] == 1:  # EV_KEY
                name = _KEYMAP.get(e["code"])
                if name:
                    if e["value"] == 0:
                        held[name] = 0.0
                    elif e["value"] == 1:
                        held[name] = 1.0
                    # value == 2 — автоповтор, состояние не меняет
            elif e["type"] == 2:  # EV_REL
                if e["code"] == REL_X:
                    dx += e["value"]
                elif e["code"] == REL_Y:
                    dy += e["value"]
            j += 1

        for k, name in enumerate(ACTION_NAMES):
            actions[i, k] = held[name]
        if mouse_mode == "discrete":
            direction, magnitude = mouse_to_discrete(dx, dy)
            actions[i, N_KEYS] = float(direction)
            actions[i, N_KEYS + 1] = magnitude
        else:
            actions[i, N_KEYS] = np.clip(dx / 50.0, -4, 4)
            actions[i, N_KEYS + 1] = np.clip(dy / 50.0, -4, 4)

    return actions


def iter_frames(episode: Path, size: tuple[int, int], color: bool = False):
    """Генератор кадров из video.mp4. Не держит видео в памяти."""
    w, h = size
    video = episode / "video.mp4"
    meta = json.loads((episode / "meta.json").read_text())
    src_w, src_h = meta["size"]

    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None

    # индексы для nearest-neighbor уменьшения
    ys = (np.arange(h) * src_h / h).astype(np.int32)
    xs = (np.arange(w) * src_w / w).astype(np.int32)

    frame_bytes = src_w * src_h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(src_h, src_w, 3)
            if color:
                yield rgb[ys][:, xs]
            else:
                gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)
                yield gray[ys][:, xs].astype(np.uint8)
    finally:
        proc.stdout.close()
        proc.wait()


def decode_frames(episode: Path, size: tuple[int, int],
                  color: bool = False) -> np.ndarray:
    """Все кадры разом (только для маленьких эпизодов)."""
    frames = list(iter_frames(episode, size, color))
    if not frames:
        shape = (0, size[1], size[0], 3) if color else (0, size[1], size[0])
        return np.empty(shape, dtype=np.uint8)
    return np.stack(frames)


def stack_frames(frames: np.ndarray, stack: int) -> np.ndarray:
    """(N, H, W) -> (N, stack, H, W), первые кадры дублируются."""
    n = len(frames)
    idx = np.arange(n)[:, None] - np.arange(stack - 1, -1, -1)[None, :]
    idx = np.clip(idx, 0, n - 1)
    return frames[idx]


def build_stream(episode: Path, out: Path, size, stack: int, shift: int,
                 color: bool, mouse_mode: str = "delta") -> Path:
    """Пишет states на диск по кадру. Нужен для больших эпизодов: RAM не растёт."""
    from collections import deque

    times = load_times(episode)
    actions = build_actions(times, load_events(episode), mouse_mode)
    n = len(times)
    channels = stack * 3 if color else stack
    h, w = size[1], size[0]

    out.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        out / "states.npy", mode="w+", dtype=np.uint8, shape=(n, channels, h, w)
    )

    buf: deque = deque(maxlen=stack)
    wrote = 0
    for i, frame in enumerate(iter_frames(episode, size, color)):
        if i >= n:
            break
        buf.append(frame)
        while len(buf) < stack:
            buf.appendleft(frame)
        stacked = np.stack(buf)
        if color:
            stacked = stacked.transpose(0, 3, 1, 2).reshape(channels, h, w)
        states[i] = stacked
        wrote += 1

    if wrote < n:
        states[wrote:] = 0
    states.flush()
    del states

    if shift:
        actions = np.concatenate(
            [actions[shift:], np.repeat(actions[-1:], shift, axis=0)], axis=0
        )
    np.save(out / "actions.npy", actions[:n])
    np.save(out / "times.npy", times - times[0])

    meta = json.loads((episode / "meta.json").read_text())
    src_size = meta.get("size")  # размер исходного видео, до уменьшения
    meta.update({"size": list(size), "src": src_size, "stack": stack,
                 "samples": n, "color": color, "action_shift": shift,
                 "mouse_mode": mouse_mode, "action_names": ACTION_NAMES})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"{episode.name}: {n} сэмплов → {out} "
          f"({n * channels * h * w / 1e9:.2f} ГБ на диске)")
    return out


def build(episode: Path, out: Path | None = None, size=(96, 54), stack: int = 4,
          shift: int = 0, color: bool = False, npy: bool = False,
          mouse_mode: str = "delta") -> Path:
    """shift — сдвиг действий относительно кадров (по умолчанию 0).

    Замер корреляции «движение мыши ↔ изменение кадра» даёт пик на +3 кадрах
    (≈200 мс). Это похоже на задержку рендера/захвата: движение мыши видно
    в кадре через ~3 кадра. Для BC это не обязательно ошибка выравнивания —
    при деплое та же задержка присутствует. Поэтому 0 по умолчанию,
    а вариант shift=3 проверяем как гипотезу на val-метриках.
    """
    if npy:
        target = out or episode / "dataset_dir"
        return build_stream(episode, target, size, stack, shift, color, mouse_mode)

    times = load_times(episode)
    events = load_events(episode)
    actions = build_actions(times, events, mouse_mode)

    frames = decode_frames(episode, size, color=color)
    n = min(len(frames), len(times))
    if len(frames) != len(times):
        print(f"[warn] кадров в видео {len(frames)}, в логе {len(times)} — берём {n}")

    states = stack_frames(frames[:n], stack)
    if color:
        # (N, stack, H, W, 3) -> (N, stack*3, H, W)
        states = states.transpose(0, 1, 4, 2, 3).reshape(n, stack * 3, *size[::-1])
    actions = actions[:n]
    if shift:
        # state[i] ↔ action[i + shift]
        actions = np.concatenate(
            [actions[shift:], np.repeat(actions[-1:], shift, axis=0)], axis=0
        )
    t = times[:n] - times[0]

    meta = json.loads((episode / "meta.json").read_text())
    src_size = meta.get("size")
    meta.update({"size": list(size), "src": src_size, "stack": stack, "samples": n,
                 "color": color, "action_shift": shift,
                 "mouse_mode": mouse_mode, "action_names": ACTION_NAMES})

    out = out or episode / ("dataset_dir" if npy else "dataset.npz")
    if npy:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "states.npy", states)
        np.save(out / "actions.npy", actions)
        np.save(out / "times.npy", t)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    else:
        np.savez_compressed(out, states=states, actions=actions, times=t,
                            meta=np.array(json.dumps(meta)))

    print(f"{episode.name}: {n} сэмплов, states {states.shape} "
          f"({states.nbytes / 1e6:.0f} МБ), actions {actions.shape}")
    print("доли удержания клавиш:")
    for k, name in enumerate(ACTION_NAMES):
        print(f"  {name:6s} {actions[:, k].mean():.2f}")
    print(f"  mouse  |dx|={np.abs(actions[:, N_KEYS]).mean():.2f} "
          f"|dy|={np.abs(actions[:, N_KEYS + 1]).mean():.2f}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=54)
    ap.add_argument("--out", type=Path, default=None,
                    help="один файл на все эпизоды (по умолчанию dataset.npz в каждом)")
    ap.add_argument("--color", action="store_true", help="хранить RGB вместо серого")
    ap.add_argument("--npy", action="store_true",
                    help="сохранить в папку .npy (для mmap, без сжатия)")
    ap.add_argument("--mouse", default="delta", choices=["delta", "discrete"],
                    help="представление мыши: дельты или 9 направлений")
    args = ap.parse_args()

    if args.out and len(args.episodes) > 1:
        raise SystemExit("--out поддерживается только для одного эпизода")

    for ep in args.episodes:
        build(ep.expanduser(), out=args.out, size=(args.width, args.height),
              stack=args.stack, shift=args.shift, color=args.color, npy=args.npy,
              mouse_mode=args.mouse)
