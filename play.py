"""Запуск обученной политики в игре.

Осторожно: это единственный скрипт, который реально управляет персонажем.
Запускать только на тестовом аккаунте. Аварийный стоп — F8 или любая
физическая клавиша (агент считает вмешательством человека и останавливается).

Использование:
  .venv/bin/python play.py checkpoints/policy.pt --dry-run   # только показать действия
  .venv/bin/python play.py checkpoints/policy.pt            # реально играть
"""

from __future__ import annotations

import argparse
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "harness"))

import capture  # noqa: E402
import uinput as vin  # noqa: E402
from record import Recorder  # noqa: E402
from train import Policy  # noqa: E402

KEY_NAMES = ["w", "a", "s", "d", "q", "e", "space", "ctrl", "shift"]
# порядок как в dataset.py: вправо, вниз-вправо, вниз, ... и «не двигать»
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
N_KEYS = 12

EVENT_FMT = "<qqHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)
KEY_F8 = 66


def _nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    sh, sw = img.shape[:2]
    ys = (np.arange(h) * sh / h).astype(np.int32)
    xs = (np.arange(w) * sw / w).astype(np.int32)
    return img[ys][:, xs]


def preprocess(frame: np.ndarray, size=(96, 54), src=(400, 224),
               color: bool = False) -> np.ndarray:
    """Повторяет цепочку dataset.py: экран → src (как писалось видео) → size."""
    small = _nearest(frame, src[0], src[1])
    small = _nearest(small, size[0], size[1])
    if not color:
        small = (small[..., 0] * 0.299 + small[..., 1] * 0.587
                 + small[..., 2] * 0.114).astype(np.uint8)
    return small.astype(np.uint8)


def game_focused() -> bool:
    """Защита от ввода в терминал: игра должна быть активным окном."""
    try:
        out = subprocess.run(["hyprctl", "activewindow", "-j"],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return True  # не смогли проверить — не блокируем
    low = out.lower()
    return "sober" in low or "roblox" in low


def focus_game() -> bool:
    """Пробуем сами переключить фокус на окно Sober."""
    try:
        # Hyprland 0.57+: dispatch переехал на Lua, старый синтаксис не работает
        subprocess.run(["hyprctl", "eval",
                        'hl.dsp.focus({window="class:org.vinegarhq.Sober"})'],
                       capture_output=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return False
    time.sleep(0.3)
    return game_focused()


def find_human_keyboards() -> list[tuple[str, int]]:
    """Физические клавиатуры и мыши: (путь, fd). Нужны и для стопа, и для DAgger."""
    found = []
    for entry in sorted(Path("/dev/input").glob("event*")):
        try:
            fd = os.open(entry, os.O_RDONLY | os.O_NONBLOCK)
        except (PermissionError, OSError):
            continue
        try:
            import fcntl
            keys = bytearray(96)
            rels = bytearray(8)
            fcntl.ioctl(fd, (2 << 30) | (96 << 16) | (ord("E") << 8) | 0x21, keys, True)
            fcntl.ioctl(fd, (2 << 30) | (8 << 16) | (ord("E") << 8) | 0x22, rels, True)
            has_mouse = rels[0] & 1 and rels[0] & 2
            if has_mouse or (keys[30 // 8] & (1 << (30 % 8))):  # REL_X/Y или KEY_A
                found.append((str(entry), fd))
            else:
                os.close(fd)
        except OSError:
            os.close(fd)
    return found


KEY_ESC = 1


def read_events(fds: list[tuple[str, int]]) -> list[tuple[str, int, int, int]]:
    """Все события с физических устройств: (путь, type, code, value)."""
    out = []
    for path, fd in fds:
        try:
            data = os.read(fd, EVENT_SIZE * 64)
        except BlockingIOError:
            continue
        for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
            _, _, etype, code, value = struct.unpack_from(EVENT_FMT, data, off)
            out.append((path, etype, code, value))
    return out


class SearchAssist:
    """Если агент полторы секунды ничего не делает, он застревает.

    Простое правило: короткий манёвр с поворотом камеры и бустом, чтобы выйти
    из пустой улицы или кустов. Не заменяет политику, только толкает её.
    """

    def __init__(self, after: float = 1.5, burst: float = 0.6):
        self.after = after
        self.burst = burst
        self.last_active = time.monotonic()
        self.until = 0.0
        self.direction = 1

    def update(self, pressed: list[str], now: float) -> tuple[list[str], tuple[int, int] | None]:
        if any(k in pressed for k in ("q", "e", "lmb")):
            self.last_active = now
        if now < self.until:
            return ["w", "space"], (self.direction * 45, 0)
        if now - self.last_active > self.after:
            self.until = now + self.burst
            self.direction *= -1
            self.last_active = now
            return ["w", "space"], (self.direction * 45, 0)
        return pressed, None


class Agent:
    def __init__(self, ckpt: Path, dry_run: bool, fps: float = 15.0):
        # оставляем ядра игре и захвату: инференс маленький, потоки только мешают
        torch.set_num_threads(2)
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.stack = data.get("stack", 4)
        meta = data.get("meta", {})
        self.size = tuple(meta.get("size", [96, 54]))  # (w, h)
        self.src = tuple(meta.get("src", [400, 224]))
        self.color = bool(meta.get("color", False))
        n_frames = self.stack // 3 if self.color else self.stack
        self.n_frames = n_frames
        self.gru = bool(data.get("gru", False))
        self.mouse_mode = data.get("mouse_mode", "delta")
        if self.gru:
            from train import PolicyGRU
            self.model = PolicyGRU(stack=self.stack, mouse_mode=self.mouse_mode)
            self.h = None
        else:
            self.model = Policy(stack=self.stack, mouse_mode=self.mouse_mode)
        self.model.load_state_dict(data["model"])
        self.model.eval()
        frame_shape = (self.size[1], self.size[0], 3) if self.color else (self.size[1], self.size[0])
        self.buf = np.zeros((n_frames, *frame_shape), dtype=np.uint8)
        self.dry_run = dry_run
        self.fps = fps
        self.vi = None if dry_run else vin.VirtualInput("aotr-agent")
        self.held: dict[str, bool] = {n: False for n in KEY_NAMES + ["r"]}
        self.btn = {"left": False, "right": False}
        self.smooth_mouse = np.zeros(2, dtype=np.float32)

    def step(self, gray: np.ndarray, mouse_scale: float = 1.0,
             thresholds: dict[str, float] | None = None) -> tuple[list[str], tuple[int, int], float]:
        t0 = time.perf_counter()
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = gray

        if self.color:
            # (frames, H, W, 3) -> (frames*3, H, W), как в dataset.py
            arr = self.buf.transpose(0, 3, 1, 2).reshape(-1, *self.size[::-1])
        else:
            arr = self.buf
        x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            if self.gru:
                logits, mouse, self.h = self.model(x.unsqueeze(1), self.h)
                logits = logits[:, -1]
                mouse = mouse[:, -1]
            else:
                logits, mouse = self.model(x)
        probs = torch.sigmoid(logits)[0].numpy()
        if self.mouse_mode == "discrete":
            vec = mouse[0]
            idx = int(vec[:9].argmax())
            mag = float(torch.sigmoid(vec[9]))
            if idx == 8:
                dx = dy = 0.0
            else:
                vx, vy = DIRS[idx]
                norm = (vx * vx + vy * vy) ** 0.5
                dx = vx / norm * mag * 1.2
                dy = vy / norm * mag * 1.2
        else:
            dx, dy = mouse[0].numpy()

        th = thresholds or {}
        pressed = [n for i, n in enumerate(KEY_NAMES)
                   if probs[i] > th.get(n, 0.5)]
        if probs[9] > th.get("lmb", 0.5):
            pressed.append("lmb")
        if probs[10] > th.get("rmb", 0.5):
            pressed.append("rmb")
        if probs[11] > th.get("r", 0.5):
            pressed.append("r")

        # без ограничения камера улетает в небо: 200 px за кадр при 15 Гц это 3000 px/с.
        # плюс сглаживание, иначе каждый кадр дёргает камеру в новую сторону
        raw = np.array([dx, dy], dtype=np.float32) * 50 * mouse_scale
        self.smooth_mouse = 0.5 * self.smooth_mouse + 0.5 * raw
        self.smooth_mouse[np.abs(self.smooth_mouse) < 2] = 0
        mdx, mdy = (int(v) for v in np.clip(self.smooth_mouse, -60, 60))
        latency = (time.perf_counter() - t0) * 1000
        return pressed, (mdx, mdy), latency

    def apply(self, pressed: list[str], mouse: tuple[int, int]) -> None:
        if self.dry_run or not self.vi:
            return
        keys = set(pressed)
        for name in self.held:
            want = name in keys
            if want != self.held[name]:
                (self.vi.key_down if want else self.vi.key_up)(name)
                self.held[name] = want
        for btn in ("left", "right"):
            want = btn in keys
            if want != self.btn[btn]:
                self.vi.hold_button(btn, want)
                self.btn[btn] = want
        if mouse != (0, 0):
            self.vi.move(*mouse)

    def release_all(self) -> None:
        if self.dry_run or not self.vi:
            return
        for name, down in self.held.items():
            if down:
                self.vi.key_up(name)
                self.held[name] = False
        for btn, down in self.btn.items():
            if down:
                self.vi.hold_button(btn, False)
                self.btn[btn] = False

    def close(self) -> None:
        self.release_all()
        if self.vi:
            self.vi.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="не отправлять ввод")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--mouse-scale", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0, help="0 = до остановки")
    ap.add_argument("--force", action="store_true", help="игнорировать проверку фокуса")
    ap.add_argument("--snap", type=Path, default=None,
                    help="куда сохранять кадры раз в 2 секунды (для разбора поведения)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="порог срабатывания клавиш (ниже — агент активнее)")
    ap.add_argument("--stop-on-key", action="store_true",
                    help="останавливать по любой клавише (по умолчанию только F8 и Esc)")
    ap.add_argument("--dagger", type=Path, default=None,
                    help="папка для записей DAgger: F8 переключает агент/человек")
    ap.add_argument("--search", action="store_true",
                    help="включать поисковый манёвр, если агент замер")
    ap.add_argument("--search-after", type=float, default=1.5,
                    help="через сколько секунд простоя начинать манёвр")
    args = ap.parse_args()

    if not args.dry_run and not args.force:
        if not game_focused():
            focus_game()
        if not game_focused():
            print("Жду фокус на игре: кликни по окну Roblox (30 секунд)...", flush=True)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not game_focused():
                time.sleep(0.5)
            if not game_focused():
                print("Игра так и не в фокусе — ввод уйдёт в другое окно. "
                      "Запусти с --force, если уверен.", file=sys.stderr)
                return 2
        print("фокус на игре — стартую", flush=True)

    human = find_human_keyboards()
    if not human:
        print("нет доступа к /dev/input — аварийный стоп по клавише не сработает",
              file=sys.stderr)

    agent = Agent(args.checkpoint, args.dry_run, args.fps)
    dagger = None
    if args.dagger:
        args.dagger.mkdir(parents=True, exist_ok=True)
        dagger = Recorder(args.dagger, fps=int(args.fps), tag="dagger")
    human_mode = False

    search = SearchAssist(after=args.search_after) if args.search else None
    if dagger:
        print("агент готов. F8 — забрать управление / вернуть агенту, Esc — выход")
    else:
        print("агент готов. F8 или Esc — стоп")

    if args.snap:
        args.snap.mkdir(parents=True, exist_ok=True)
    last_snap = 0.0

    period = 1.0 / args.fps
    next_t = time.monotonic()
    t_start = time.monotonic()
    n = 0
    ticks = 0
    paused = False
    lat_sum = 0.0
    thresholds = {k: args.threshold for k in KEY_NAMES + ["lmb", "rmb", "r"]}
    for k in ("ctrl", "shift", "rmb", "r"):  # редкие клавиши не должны срабатывать случайно
        thresholds[k] = max(args.threshold, 0.7)
    try:
        while True:
            now = time.monotonic()
            if args.seconds and now - t_start > args.seconds:
                break
            for path, etype, code, value in read_events(human):
                if etype == 1 and value == 1 and code == KEY_ESC:
                    raise KeyboardInterrupt
                if etype == 1 and value == 1 and code == KEY_F8:
                    if dagger:
                        human_mode = not human_mode
                        if human_mode:
                            agent.release_all()
                            dagger.start()
                            print("[dagger] играет человек, F8 — вернуть агенту", flush=True)
                        else:
                            dagger.stop()
                            if getattr(agent, "gru", False):
                                agent.h = None
                            agent.smooth_mouse[:] = 0
                            print("[dagger] агент продолжает", flush=True)
                    else:
                        raise KeyboardInterrupt
                    continue
                if human_mode and dagger:
                    dagger.log_event(path, time.monotonic(), etype, code, value)
                elif (args.stop_on_key and etype == 1 and value == 1
                      and code < 0x100):
                    raise KeyboardInterrupt

            # без фокуса не шлём ввод (иначе он уйдёт в терминал), но и не выходим
            if not args.dry_run and ticks % 30 == 0:
                focused = game_focused()
                if not focused and not paused:
                    paused = True
                    agent.release_all()
                    print("фокус ушёл — пауза, вернись в игру", flush=True)
                elif focused and paused:
                    paused = False
                    if getattr(agent, "gru", False):
                        agent.h = None
                    print("фокус вернулся — продолжаю", flush=True)
            if paused:
                time.sleep(0.02)
                ticks += 1
                continue
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue

            if human_mode and dagger:
                ticks += 1
                dagger.grab_frame()
                next_t += period
                if next_t < time.monotonic():
                    next_t = time.monotonic() + period
                continue

            frame = capture.grab()
            gray = preprocess(frame, agent.size, agent.src, agent.color)
            pressed, mouse, lat = agent.step(gray, args.mouse_scale, thresholds)
            if search:
                pressed, override = search.update(pressed, time.monotonic())
                if override is not None:
                    mouse = override
            agent.apply(pressed, mouse)
            n += 1
            lat_sum += lat
            ticks += 1
            if args.snap and time.monotonic() - last_snap >= 2.0:
                subprocess.Popen(["grim", "-t", "png", str(args.snap / f"{n:06d}.png")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                last_snap = time.monotonic()
            if n % 15 == 0:
                print(f"{n / (time.monotonic() - t_start + 1e-9):.1f} fps, "
                      f"инференс {lat_sum / n:.1f} мс, "
                      f"клавиши {','.join(pressed) or '-'}", flush=True)
            next_t += period
            if next_t < time.monotonic():
                next_t = time.monotonic() + period
    except KeyboardInterrupt:
        pass
    finally:
        if dagger and dagger.recording:
            dagger.stop()
        agent.close()
        for _, fd in human:
            os.close(fd)
    print(f"остановлено, кадров {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
