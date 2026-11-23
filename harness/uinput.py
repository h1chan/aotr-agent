"""Виртуальная клавиатура и мышь через /dev/uinput.

Зачем: агент должен отправлять ввод в игру так, будто это делает человек.
uinput создаёт настоящее устройство на уровне ядра — приложение не может
отличить его от физической клавиатуры/мыши.

Требования: доступ на запись к /dev/uinput (ACL уже выдан пользователю h1chan).
Root не нужен.

Проверка: python3 uinput.py --selftest
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import time

# --- ioctl -----------------------------------------------------------------

UI_DEV_SETUP = 0x405C5503     # _IOW('U', 3, struct uinput_setup), size=92
UI_DEV_DESTROY = 0x5502       # _IO('U', 2)
UI_SET_EVBIT = 0x40045564     # _IOW('U', 100, int)
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566

# --- event types -----------------------------------------------------------

EV_SYN, EV_KEY, EV_REL = 0x00, 0x01, 0x02
SYN_REPORT = 0x00

REL_X, REL_Y, REL_WHEEL = 0x00, 0x01, 0x08
BTN_LEFT, BTN_RIGHT, BTN_MIDDLE = 0x110, 0x111, 0x112

# --- key codes (linux/input-event-codes.h) ---------------------------------

KEYS = {
    "esc": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "tab": 15, "q": 16, "w": 17, "e": 18, "r": 19,
    "enter": 28, "ctrl": 29, "a": 30, "s": 31, "d": 32,
    "shift": 42, "b": 48, "space": 57, "f": 33, "m": 50,
}


class _InputId(ctypes.Structure):
    _fields_ = [
        ("bustype", ctypes.c_uint16),
        ("vendor", ctypes.c_uint16),
        ("product", ctypes.c_uint16),
        ("version", ctypes.c_uint16),
    ]


class _UinputSetup(ctypes.Structure):
    _fields_ = [
        ("id", _InputId),
        ("name", ctypes.c_char * 80),
        ("ff_effects_max", ctypes.c_uint32),
    ]


class InputEvent(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long),
        ("type", ctypes.c_uint16),
        ("code", ctypes.c_uint16),
        ("value", ctypes.c_int32),
    ]


class VirtualInput:
    """Одно uinput-устройство: клавиатура + относительная мышь."""

    def __init__(self, name: str = "aotr-agent"):
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        self._setup(name)

    def _setup(self, name: str) -> None:
        for ev in (EV_KEY, EV_REL, EV_SYN):
            fcntl.ioctl(self.fd, UI_SET_EVBIT, ev)

        for code in KEYS.values():
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
        for code in (BTN_LEFT, BTN_RIGHT, BTN_MIDDLE):
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
        for code in (REL_X, REL_Y, REL_WHEEL):
            fcntl.ioctl(self.fd, UI_SET_RELBIT, code)

        setup = _UinputSetup()
        setup.id.bustype = 0x03          # BUS_USB
        setup.id.vendor = 0x1234
        setup.id.product = 0x5678
        setup.id.version = 1
        setup.name = name.encode()[:79]
        fcntl.ioctl(self.fd, UI_DEV_SETUP, setup)
        fcntl.ioctl(self.fd, 0x5501)     # UI_DEV_CREATE
        time.sleep(0.1)                  # дать ядру создать устройство

    # --- низкий уровень ---

    def _emit(self, ev_type: int, code: int, value: int) -> None:
        ev = InputEvent(0, 0, ev_type, code, value)
        os.write(self.fd, bytes(ev))

    def _sync(self) -> None:
        self._emit(EV_SYN, SYN_REPORT, 0)

    # --- клавиатура ---

    def key_down(self, key: str) -> None:
        self._emit(EV_KEY, KEYS[key], 1)
        self._sync()

    def key_up(self, key: str) -> None:
        self._emit(EV_KEY, KEYS[key], 0)
        self._sync()

    def tap(self, key: str, hold: float = 0.03) -> None:
        self.key_down(key)
        time.sleep(hold)
        self.key_up(key)

    def keys_down(self, *keys: str) -> None:
        for k in keys:
            self._emit(EV_KEY, KEYS[k], 1)
        self._sync()

    def keys_up(self, *keys: str) -> None:
        for k in keys:
            self._emit(EV_KEY, KEYS[k], 0)
        self._sync()

    # --- мышь ---

    def move(self, dx: int, dy: int) -> None:
        if dx:
            self._emit(EV_REL, REL_X, int(dx))
        if dy:
            self._emit(EV_REL, REL_Y, int(dy))
        self._sync()

    def click(self, button: str = "left") -> None:
        code = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}[button]
        self._emit(EV_KEY, code, 1)
        self._sync()
        self._emit(EV_KEY, code, 0)
        self._sync()

    def hold_button(self, button: str = "left", pressed: bool = True) -> None:
        code = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}[button]
        self._emit(EV_KEY, code, 1 if pressed else 0)
        self._sync()

    def scroll(self, amount: int) -> None:
        self._emit(EV_REL, REL_WHEEL, int(amount))
        self._sync()

    # --- жизненный цикл ---

    def close(self) -> None:
        if self.fd >= 0:
            fcntl.ioctl(self.fd, UI_DEV_DESTROY)
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "VirtualInput":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    import sys

    with VirtualInput("aotr-agent-selftest") as vi:
        print("устройство создано: /dev/input/", end="")
        # найти только что созданный узел
        for entry in sorted(os.listdir("/dev/input"), reverse=True):
            if entry.startswith("event"):
                try:
                    with open(f"/sys/class/input/{entry}/device/name") as f:
                        if "aotr-agent" in f.read():
                            print(entry)
                            break
                except OSError:
                    pass
        if "--move" in sys.argv:
            print("двигаю курсор на 20px вправо и обратно")
            vi.move(20, 0)
            time.sleep(0.2)
            vi.move(-20, 0)
    print("устройство удалено")
