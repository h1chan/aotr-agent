"""Обучение политики behavior cloning на собранных эпизодах.

Вход: dataset.npz из dataset.py (states, actions, times).
Модель: маленькая CNN (4 conv + fc), две головы — клавиши (BCE) и мышь (SmoothL1).

Использование:
  .venv/bin/python train.py ~/aotr-data/episode_0001 [ещё эпизоды...] --epochs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N_KEYS = 12  # первые 12 колонок actions — клавиши, последние 2 — мышь


class PolicyGRU(nn.Module):
    """Та же CNN, но с рекуррентной памятью между кадрами.

    CNN смотрит на стек кадров, GRU держит скрытое состояние: модель помнит,
    что было несколько секунд назад, даже если объект вышел из кадра.
    """

    def __init__(self, stack: int = 12, hidden: int = 256, n_keys: int = N_KEYS,
                 mouse_mode: str = "delta"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(stack, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3 * 5, 256),
                                   nn.ReLU(inplace=True))
        self.gru = nn.GRU(256, hidden, batch_first=True)
        self.keys = nn.Linear(hidden, n_keys)
        self.mouse = nn.Linear(hidden, 10 if mouse_mode == "discrete" else 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        # x: (B, T, C, H, W)
        b, t = x.shape[:2]
        f = self.trunk(self.features(x.reshape(b * t, *x.shape[2:])))
        out, h = self.gru(f.reshape(b, t, -1), h)
        return self.keys(out), self.mouse(out), h


class SequenceDataset(torch.utils.data.Dataset):
    """Куски по seq кадров внутри одного эпизода, без перехода через границу train/val."""

    def __init__(self, sources, seq: int, train: bool = True):
        self.sources = sources
        self.seq = seq
        self.items: list[tuple[int, int]] = []
        for si, (_, _, tr, va) in enumerate(sources):
            idx = tr if train else va
            if len(idx) < seq:
                continue
            start, end = int(idx[0]), int(idx[-1]) + 1
            for s in range(start, end - seq + 1, seq):
                self.items.append((si, s))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        si, s = self.items[i]
        x, y, _, _ = self.sources[si]
        return (torch.from_numpy(np.array(x[s:s + self.seq], dtype=np.uint8, copy=True)),
                torch.from_numpy(np.array(y[s:s + self.seq], dtype=np.float32, copy=True)))


class Policy(nn.Module):
    def __init__(self, stack: int = 4, n_keys: int = N_KEYS,
                 mouse_mode: str = "delta"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(stack, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3 * 5, 256), nn.ReLU(inplace=True))
        self.keys = nn.Linear(256, n_keys)
        self.mouse = nn.Linear(256, 10 if mouse_mode == "discrete" else 2)
        nn.init.zeros_(self.mouse.weight)
        nn.init.zeros_(self.mouse.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.features(x))
        return self.keys(h), self.mouse(h)


class FrameDataset(torch.utils.data.Dataset):
    """Несколько эпизодов без склейки в один массив: mmap остаётся на диске."""

    def __init__(self, sources: list[tuple]):
        # sources: список (states, actions, indices)
        self.sources = sources
        self.index: list[tuple[int, int]] = []
        for si, (_, _, idxs) in enumerate(sources):
            self.index.extend((si, int(j)) for j in idxs)
        self.shape = sources[0][0].shape[1:]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        si, j = self.index[i]
        states, actions, _ = self.sources[si]
        x = np.array(states[j], dtype=np.uint8, copy=True)
        y = np.array(actions[j], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)

    def actions_for(self, which: int = 0) -> np.ndarray:
        """Все целевые действия выбранного источника (для pos_weight)."""
        _, actions, idxs = self.sources[which]
        return actions[idxs]


def load_episodes(paths: list[Path], val_frac: float = 0.15, repeat_last: int = 1):
    """Пути могут быть .npz или папками с states.npy (mmap, не грузим в RAM)."""
    # если весь датасет влезает в память, грузим целиком: mmap на случайном
    # доступе читает с диска каждый батч и замедляет эпоху в разы
    total_bytes = sum((p / "states.npy").stat().st_size for p in paths if p.is_dir())
    in_ram = total_bytes < 2.5e9  # выше — OOM на 7.4 ГБ RAM
    if total_bytes:
        print(f"датасет {total_bytes / 1e9:.2f} ГБ, режим: "
              f"{'в памяти' if in_ram else 'mmap с диска'}")

    sets = []
    for p in paths:
        if p.is_dir():
            x = np.load(p / "states.npy", mmap_mode=None if in_ram else "r")
            y = np.load(p / "actions.npy").astype(np.float32)
            name = p.name
        else:
            d = np.load(p)
            x = d["states"]
            y = d["actions"].astype(np.float32)
            name = p.stem
        n = len(x)
        cut = int(n * (1 - val_frac))
        sets.append((x, y, np.arange(cut), np.arange(cut, n)))
        print(f"{name}: {n} сэмплов, train {cut}, val {n - cut}")

    if len({s[0].shape[1:] for s in sets}) != 1:
        raise SystemExit("у эпизодов разная форма входа, собери их с одинаковыми --width/--height")

    if repeat_last > 1:  # DAgger-исправления весим больше обычных кадров
        x, y, tr, va = sets[-1]
        sets[-1] = (x, y, np.tile(tr, repeat_last), va)
        print(f"последний источник повторён ×{repeat_last}")

    train = FrameDataset([(x, y, tr) for x, y, tr, _ in sets])
    val = FrameDataset([(x, y, va) for x, y, _, va in sets])
    return train, val


def load_sources(paths: list[Path], val_frac: float = 0.15):
    """Как load_episodes, но возвращает сырые массивы: нужно для SequenceDataset."""
    total_bytes = sum((p / "states.npy").stat().st_size for p in paths if p.is_dir())
    in_ram = total_bytes < 2.5e9  # выше — OOM на 7.4 ГБ RAM
    if total_bytes:
        print(f"датасет {total_bytes / 1e9:.2f} ГБ, режим: "
              f"{'в памяти' if in_ram else 'mmap с диска'}")
    sources = []
    for p in paths:
        if p.is_dir():
            x = np.load(p / "states.npy", mmap_mode=None if in_ram else "r")
            y = np.load(p / "actions.npy").astype(np.float32)
            name = p.name
        else:
            d = np.load(p)
            x, y, name = d["states"], d["actions"].astype(np.float32), p.stem
        n = len(x)
        cut = int(n * (1 - val_frac))
        sources.append((x, y, np.arange(cut), np.arange(cut, n)))
        print(f"{name}: {n} сэмплов, train {cut}, val {n - cut}")
    if len({tuple(s[0].shape[1:]) for s in sources}) != 1:
        raise SystemExit("у эпизодов разная форма входа, собери их с одинаковыми --width/--height")
    return sources


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, bce: nn.Module, sl1: nn.Module,
             gru: bool = False, device=None, mouse_mode: str = "delta") -> dict:
    device = device or torch.device('cpu')
    model.eval()
    total_bce = total_mse = 0.0
    n = 0
    tp = fp = fn = 0
    per_key_tp = np.zeros(N_KEYS)
    per_key_pos = np.zeros(N_KEYS)
    dir_hits = dir_total = 0
    mag_err = mag_n = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float().div_(255.0)
        yb = yb.to(device)
        logits, mouse = model(xb)[:2]
        if gru:
            logits = logits.reshape(-1, N_KEYS)
            mouse = mouse.reshape(-1, 2)
            yb = yb.reshape(-1, yb.shape[-1])
        total_bce += bce(logits, yb[:, :N_KEYS]).item() * len(xb)
        if mouse_mode == "discrete":
            dir_logits, mag_logit = mouse[:, :9], mouse[:, 9]
            mag_err += (torch.sigmoid(mag_logit) - yb[:, N_KEYS + 1]).abs().sum().item()
            dir_hits += int((dir_logits.argmax(1) == yb[:, N_KEYS].long()).sum())
            dir_total += len(xb)
            total_mse += 0.0
        else:
            total_mse += sl1(mouse, yb[:, N_KEYS:]).item() * len(xb)
        pred = (torch.sigmoid(logits) > 0.5).float()
        yk = yb[:, :N_KEYS]
        tp += int(((pred == 1) & (yk == 1)).sum())
        fp += int(((pred == 1) & (yk == 0)).sum())
        fn += int(((pred == 0) & (yk == 1)).sum())
        per_key_tp += ((pred == 1) & (yk == 1)).sum(dim=0).numpy()
        per_key_pos += (yk == 1).sum(dim=0).numpy()
        n += len(xb)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    recall_per_key = per_key_tp / np.maximum(per_key_pos, 1)
    result = {"bce": total_bce / n, "mouse_l1": total_mse / n,
              "precision": precision, "recall": recall, "f1": f1,
              "recall_per_key": recall_per_key}
    if mouse_mode == "discrete":
        result["dir_acc"] = dir_hits / max(1, dir_total)
        result["mag_mae"] = mag_err / max(1, dir_total)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/policy.pt"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--resume", type=Path, default=None,
                    help="продолжить с чекпоинта")
    ap.add_argument("--mouse", default="auto", choices=["auto", "delta", "discrete"],
                    help="как предсказывать мышь; auto берёт из meta.json")
    ap.add_argument("--repeat-last", type=int, default=1,
                    help="во сколько раз повторить train последнего источника")
    ap.add_argument("--gru", action="store_true", help="рекуррентная память (GRU)")
    ap.add_argument("--seq", type=int, default=12, help="длина последовательности для GRU")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)

    if args.gru and args.repeat_last > 1:
        print("предупреждение: --repeat-last не применяется в режиме --gru")
    mouse_mode = args.mouse
    if mouse_mode == "auto":
        first = args.episodes[0]
        if first.is_dir():
            meta = json.loads((first / "meta.json").read_text())
        else:
            meta = json.loads(str(np.load(first)["meta"]))
        mouse_mode = meta.get("mouse_mode", "delta")
    print(f"мышь: {mouse_mode}")

    if args.gru:
        sources = load_sources(args.episodes)
        train_ds = SequenceDataset(sources, args.seq, True)
        val_ds = SequenceDataset(sources, args.seq, False)
        stack = sources[0][0].shape[1]
        model = PolicyGRU(stack=stack, mouse_mode=mouse_mode)
        print(f"GRU: seq {args.seq}, train {len(train_ds)} последовательностей")
    else:
        train_ds, val_ds = load_episodes(args.episodes, repeat_last=args.repeat_last)
        stack = train_ds.shape[0]  # (C, H, W)
        model = Policy(stack=stack, mouse_mode=mouse_mode)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"продолжаю с {args.resume}")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"устройство: {device}")
    print(f"модель: {n_params / 1e3:.0f}k параметров, вход {stack} каналов, "
          f"train {len(train_ds)}, val {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    # клавиши редкие и несбалансированные: без весов сеть схлопывается в «ничего не нажато»
    if args.gru:
        pos = np.concatenate([
            y[tr][:, :N_KEYS] for _, y, tr, _ in train_ds.sources
        ], axis=0).mean(axis=0)
    else:
        pos = np.concatenate([
            acts[idxs][:, :N_KEYS] for _, acts, idxs in train_ds.sources
        ], axis=0).mean(axis=0)
    pos_weight = torch.tensor(
        np.clip((1 - pos) / np.maximum(pos, 1e-3), 1.0, 20.0), dtype=torch.float32
    )
    print("pos_weight:", {n: round(float(w), 1) for n, w in zip(
        ["w", "a", "s", "d", "q", "e", "space", "ctrl", "shift", "lmb", "rmb", "r"],
        pos_weight)})
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    sl1 = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (xb, yb) in enumerate(train_dl):
            xb = xb.to(device).float().div_(255.0)
            yb = yb.to(device)
            logits, mouse = model(xb)[:2]
            if args.gru:
                logits = logits.reshape(-1, N_KEYS)
                mouse = mouse.reshape(-1, 2)
                yb = yb.reshape(-1, yb.shape[-1])
            loss = bce(logits, yb[:, :N_KEYS])
            if mouse_mode == "discrete":
                dir_logits, mag_logit = mouse[:, :9], mouse[:, 9]
                loss = loss + nn.functional.cross_entropy(dir_logits, yb[:, N_KEYS].long())
                loss = loss + nn.functional.mse_loss(torch.sigmoid(mag_logit),
                                                     yb[:, N_KEYS + 1])
            else:
                loss = loss + sl1(mouse, yb[:, N_KEYS:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()
        m = evaluate(model, val_dl, bce, sl1, gru=args.gru, device=device,
                     mouse_mode=mouse_mode)
        rpk = " ".join(f"{n}:{r:.2f}" for n, r in zip(
            ["w", "a", "s", "d", "q", "e", "sp", "ct", "sh", "lm", "rm", "r"],
            m["recall_per_key"]) if r > 0.001)
        mouse_stats = (f"dir {m['dir_acc']:.3f} mag {m['mag_mae']:.3f} "
                       if mouse_mode == "discrete" else f"l1 {m['mouse_l1']:.4f} ")
        print(f"epoch {epoch:2d}: loss {running / max(1, len(train_dl)):.4f} | "
              f"val bce {m['bce']:.4f} {mouse_stats}"
              f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"| {time.time() - t0:.0f} c")
        print(f"          recall по клавишам: {rpk}")

        score = m["f1"] - 0.1 * m["mouse_l1"]
        if score > best:
            best = score
            first = args.episodes[0]
            meta = (json.loads((first / "meta.json").read_text()) if first.is_dir()
                    else json.loads(str(np.load(first)["meta"])))
            torch.save({"model": model.state_dict(), "stack": stack, "meta": meta,
                        "gru": args.gru, "seq": args.seq, "mouse_mode": mouse_mode,
                        "hp": {k: str(v) for k, v in vars(args).items()}}, args.out)
            print(f"  сохранено → {args.out}")
    print(f"лучший score {best:.3f}")


if __name__ == "__main__":
    main()
