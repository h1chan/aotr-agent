# Telemetry spec for a private place copy

What to log if the AoTR team provides a private `.rbxl` for research. The goal is a fast, honest training environment: state in numbers, actions, reward, and a reset.

This is a server-side logging design for a copy we control. It is not a client modification and not an executor script.

## Why

Pixel-only training needs hours of human demonstrations and still lacks a goal signal. With exact state we can:

- train a policy on numbers first, then distil it into pixels;
- reset a mission in under a second instead of minutes;
- compute reward exactly;
- generate expert data automatically once a policy is decent.

## Sampling

- 20–30 Hz, fixed timestep, one record per tick.
- One `episode_id` per mission attempt, one `tick` counter inside it.
- Log the game build/version and the place config hash in episode metadata.

## Episode metadata

| Field | Type | Notes |
|---|---|---|
| `episode_id` | string | UUID |
| `started_at` | ISO timestamp | wall clock |
| `place_version` | string | from the copy |
| `map` | string | Shiganshina, Trost, … |
| `difficulty` | string | Easy…Freedom |
| `modifiers` | list[str] | Nightmare, Oddball, … |
| `policy` | string | `human`, `scripted`, `rl-v3` |
| `seed` | int | if the mission has randomness |

## Per-tick player state

| Field | Type | Notes |
|---|---|---|
| `pos` | float[3] | world position |
| `vel` | float[3] | velocity |
| `cam_forward` | float[3] | unit vector |
| `health`, `max_health` | float | |
| `gas`, `max_gas` | float | |
| `blades`, `max_blades` | int | |
| `injuries` | int[8] | head, eyes, torso, arms, legs, gear |
| `hook_left`, `hook_right` | bool + float[3] | attached and hit point |
| `boosting` | bool | |
| `shifted` | bool | titan form |
| `rage` | float | shifter meter |
| `streak`, `level`, `xp`, `gold` | int | |

## Titans

Per tick, a list. Cap at 32 nearest, sorted by distance.

| Field | Type | Notes |
|---|---|---|
| `id` | int | stable within episode |
| `kind` | enum | pure, abnormal, crawler, ducker, leader, rager, ice |
| `pos`, `vel` | float[3] | |
| `nape_pos` | float[3] | the hit point |
| `hp`, `max_hp` | float | |
| `state` | enum | idle, chase, attack, stunned |
| `aggro_target` | int | player id or -1 |
| `distance` | float | to player |

## Mission

| Field | Type | Notes |
|---|---|---|
| `phase` | enum | start, combat, objective, end |
| `time_left` | float | |
| `kills`, `kills_target` | int | |
| `objectives` | list | type, progress, target |
| `resupply` | list[float[3]] | |
| `done` | bool | episode finished |
| `outcome` | enum | win, death, fail, timeout |

## Actions

Logged at the same rate as state, as the action applied during the tick.

| Field | Type | Notes |
|---|---|---|
| `keys` | bool[12] | w a s d q e space ctrl shift lmb rmb r |
| `mouse_dir` | int | 0–7, 8 = none |
| `mouse_mag` | float | 0–1 |
| `look_delta` | float[2] | optional raw delta for analysis |

## Reward

Computed server-side, logged per tick and summed per episode.

- `+1` per titan kill.
- `+0.1 × damage_dealt / titan_max_hp` as shaping.
- `-1` on death.
- `+small` for objective progress.
- `-0.001` per tick to discourage idling.

Log both the components and the total, so shaping can be changed offline.

## Reset

- One function `reset(seed, map, difficulty, modifiers)` that returns the first observation.
- Target: under one second from call to playable.
- Log a `reset` event with the same metadata.

## Storage

- JSONL per episode is easiest to write and debug.
- Parquet for training, one file per 100 episodes.
- Keep raw for a week, then keep only episodes used in training plus 10% random sample.

## Privacy and scope

- Only gameplay data. No account names, no chat, no Roblox user ids.
- Never log anything outside the private copy.
- If the copy is shared with us under NDA, the data stays with the project.

## What this unlocks

With this schema and a fast reset, a small MLP on state can learn a competent policy in hours, then be distilled into the pixel model we already have. That is the shortest path from "moves and looks at titans" to "farms reliably".
