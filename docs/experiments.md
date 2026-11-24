# Experiment log

Honest record of what was tried, what worked, and what did not. Numbers are from held-out splits of the recorded episodes.

## Setup

- CPU: Intel i5-7400, 4 cores, 7.4 GB RAM, no discrete GPU.
- OS: Arch Linux, Hyprland, Roblox through Sober (XWayland).
- Capture: `grim`, about 30 FPS at 1600×900, 15 FPS used for training.
- Input: `/dev/uinput`, virtual keyboard and relative mouse.
- Training: PyTorch CPU, Python 3.12.

## Data

| Episode | Map | Difficulty | Duration | Frames | FPS |
|---|---|---|---|---|---|
| 0001 | Giant Forest | Hard | 10.6 min | 9 553 | 15.0 |
| 0002 | Giant Forest | Hard | 18.6 min | 16 731 | 15.0 |
| 0003 | Shiganshina | Severe | 21.4 min | 19 221 | 15.0 |
| 0004 | Trost | Severe | 16.4 min | 14 781 | 15.0 |

Plus 12 DAgger episodes, 4.3 min total, 3 526 frames.

## Policy

| Model | Input | Mouse | Key F1 | Mouse metric |
|---|---|---|---|---|
| v0 | 96×54 gray, 4 frames | delta | 0.431 | L1 0.115 |
| hi | 160×90 gray, 6 frames | delta | 0.494 | L1 0.117 |
| color | 128×72 color, 4 frames | delta | 0.511 | L1 0.113 |
| v2 | 128×72 color, 4 frames, 26k samples | delta | 0.566 | L1 0.131 |
| gru | 128×72 color, sequences of 8 | delta | 0.444 | L1 0.139 |
| disc | 96×54 color, 4 frames, 64k samples | 9-way + magnitude | 0.531 | dir acc 0.424 |

Observations:

- Color helped more than resolution.
- Delta regression produced near-zero mouse output, correlation with human turns around zero.
- Discrete mouse reached 42% direction accuracy over 9 classes after 5 epochs, and the camera visibly turns toward titans in game.
- GRU was slower to train and did not beat the CNN at this data scale.
- The 128×72 discrete model reached 0.435 direction accuracy. On 2026-11-24 it landed the first confirmed live kill: the HUD read `Slay Titans [1/23]` about 65 seconds into the run. Reliability is still low, and the virtual mouse can click menu buttons after death.

## Titan detector

Goal: provide a goal vector (bearing, distance) to the policy.

1. **Colour heuristic.** Bright, low-texture blobs. Failed: found fog between trees and tree trunks. Titan skin is close to bark and fog.
2. **Labelled data.** 300 frames, point labels on titans, collected with a local browser tool.
3. **Models.**
   - 200×112, BCE + pos_weight: precision 0.47, recall 0.68 at threshold 0.6, median hit error 52 px.
   - 320×180, BCE: collapsed to a constant blob. Positive cells are under 0.3% of the 80×45 target.
   - 320×180, Dice loss: same collapse.
   - Mini-batch fix: the original trainer used the whole dataset as one batch, one optimizer step per epoch. Fixed, but 300 frames were still too few.
4. **Signal check.** Detector bearing correlated −0.19 with the direction the human turned the camera. The detector was looking at the wrong thing.

Decision: closed. Code kept in `detector/`. Will be revisited with more labels, a pretrained backbone (MobileNetV3), or a private place copy that provides exact positions.

## Runtime

| Component | Time |
|---|---|
| Capture 1600×900 (grim) | 30–50 ms |
| CNN inference, 96×54 | 2–6 ms |
| GRU inference | 5–7 ms |
| Full loop | 66–77 ms (13–15 FPS) |
| Dataset build, 16k frames | 40–60 s |
| CNN epoch, 64k samples | 80–135 s |

## Known issues

- Mission end screen is not handled; the agent stops there.
- Four-frame memory; targets are forgotten when they leave the frame.
- No goal input, so exploration is scripted rather than learned.
- Behaviour depends on the recorded build and gear.
