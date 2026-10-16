# K4-Track4-Capstone-TeamImback — Real-Time Fall Detection (MediaPipe Pose)

Detects potential falls from a webcam or video using the pretrained MediaPipe Pose Landmarker
and a rule-based state machine. No model training.

## Setup (Windows PowerShell; macOS/Linux: `source .venv/bin/activate`)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py     # pose models -> models/, UR Fall clips -> test_videos/
```

Model download (done by the script, or manually):
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
→ save as `models/pose_landmarker_lite.task`.

## Web demo (recommended for presenting)

```powershell
python webapp.py            # then open http://127.0.0.1:8000
```

- **Source picker:** webcam, any downloaded UR Fall clip (labelled by activity), or **upload your own video**.
- **Live view:** the annotated video, a large state badge, condition chips (fast movement / body flat / hips low / upright / pose found) and the live readouts.
- **Charts:** 15-second charts of torso and leg angle, hip speed and hip height, with the thresholds drawn on them, plus a state timeline.
- **Tabs:**
  - **Event log:** every state change.
  - **Thresholds (live):** sliders that retune the running detector instantly.
  - **Test results:** click a row to play that clip.
  - **How it works.**
- **Pose model selector:** Lite (fast) or Full (more accurate). The pill in the top-right shows which model is running.
- **Long-lie escalation:** during a fall alert, the page counts how long the person has been motionless. After `still_alert` s the alert becomes **NO MOVEMENT: CALL FOR HELP**.
- **Alert snapshots:** the frame at every fall alert and every escalation, shown in a gallery tab.
- **Benchmark with current thresholds** (Results tab): re-runs the public clips with the slider values and fills in the score clip by clip.
- An alarm sound plays during FALL_ALERT (toggle in the toolbar). `/#results`, `/#tuning` or `/#snaps` opens a tab directly.
- No extra dependencies (Python standard-library server). It binds to `127.0.0.1` only.

**Camera tip:** shoulders and hips must be in view, ideally knees and ankles too. A desk webcam that shows only
your upper body will show "NO POSE". Step back, and a side-on view works best.

## Track 4 pillars used (also in the web demo: tab "Trụ cột Track 4", `/#pillars`)

| Pillar | Strength | Where in code | What it does |
|---|---|---|---|
| Representation | strong | `fall_detector.Features`, `_features()`, `_step()` | 33 landmarks → 6 physical features in torso-lengths → 4-state machine |
| Time | strong | `main.frames/process`, `_features()` (gap, EMA), `_step()` timers, `webapp.Engine._run` | real dt, smoothing, persistence windows, gap reset, long-lie timer, real-time playback + fps |
| Uncertainty | strong | `_mid()`, `settled`, `Config.min_visibility/still_motion` | visibility gating, state frozen on pose loss, knee fallback, jitter-measured stillness threshold |
| Evidence | strong | `evaluate.py`, results files, GUI benchmark | holdout 16/21 (falls 6/10, ADL 10/11), failure cases with causes, 35 tests |
| Data | strong | `download_data.py`, `evaluate.py`, synthetic poses in tests | public UR Fall clips, labels, tune/holdout split, replay, uploads |
| Geometry | medium | `_features()` (`angle`, `leg_angle`, `ref_torso`) | pixel-space angles, scale-free normalization, leg axis for foreshortened falls; no calibration |
| Learning | weak | `main.make_landmarker()` | pretrained MediaPipe only; fall decision is rule-based |

## Run (OpenCV window)

```powershell
python main.py                                                         # webcam
python main.py --source my_video.mp4                                   # any video
python main.py --source test_videos\tune\fall-01-cam0.mp4 --crop-right  # UR Fall clip (depth|RGB)
python main.py --model models\pose_landmarker_full.task                # more accurate, slower
python evaluate.py "test_videos/holdout/*.mp4"                         # results table
python -m pytest -q                                                    # unit tests
```

Keys: `q`/`Esc` quit, `space` pause.
The overlay shows the skeleton, a state bar (green NORMAL, yellow POSSIBLE_FALL, red FALL_ALERT,
blue RECOVERY), torso angle, hip speed and hip height. A fall flashes a red frame and "POTENTIAL FALL!".

## How it works

1. MediaPipe gives 33 landmarks per frame. We use shoulders, hips, knees and ankles. A left/right pair is ignored when neither point has visibility ≥ `min_visibility`.
2. Light exponential smoothing (`smooth_alpha`) on the mid-hip, mid-shoulder, ankle and knee points.
3. Features, normalized by the person's **torso length** measured while upright, so camera distance doesn't matter:
   - **angle**: torso (mid-hip → mid-shoulder) angle from vertical. 0° is upright, 90° horizontal, 180° head-down.
   - **leg angle**: shoulder → ankle line angle from vertical. This catches falls *toward* the camera, where the torso looks short and vertical even when the person is flat.
   - **hip v**: downward hip speed in torso-lengths/s, using real timestamps.
   - **angle rate**: deg/s.
   - **hip h**: hip height above the ankles (or the knees if the ankles are hidden) in torso-lengths. This separates *bending* (hips high) from *lying* (hips on the floor), and *crawling* (knees below hips) from lying.
4. State machine:
   - NORMAL → POSSIBLE_FALL: hip v ≥ `fall_speed` **or** angle rate ≥ `angle_rate`
   - POSSIBLE_FALL → FALL_ALERT: **lying** for `alert_hold` s. Lying means the torso or leg angle is within [`horizontal_angle`, 180 − `horizontal_angle`] **and** hip h ≤ `hip_low_ratio`.
   - POSSIBLE_FALL → NORMAL: not lying within `possible_window` s (for example, sitting down fast)
   - FALL_ALERT → RECOVERY: upright (angle ≤ `upright_angle` and hips off the floor) for `recovery_hold` s
   - RECOVERY → NORMAL after `recovery_cooldown` s
   - **Long lie:** during FALL_ALERT, body motion (smoothed hip and shoulder movement, torso-lengths/s) below `still_motion` for `still_alert` s escalates the alert to "no movement". The 0.3 threshold sits above the landmark jitter of someone lying still (0.07–0.19 measured on adl-10) and below the movement people make settling after a fall (0.5–1.4).
   - No pose: the state is frozen. After a gap longer than `max_gap` s, velocity history resets, so a person reappearing somewhere else doesn't produce a fake speed spike.

Lying down **slowly** never passes the speed gate, so it doesn't alert. A jolt while already lying still (for example landmark jitter)
is ignored: a fast movement only opens POSSIBLE_FALL if the person hadn't already been lying for `alert_hold` s.

## Thresholds (`fall_detector.Config`)

| Field | Value | Meaning |
|---|---|---|
| `min_visibility` | 0.5 | landmark pairs below this count as missing |
| `smooth_alpha` | 0.5 | EMA weight of the newest frame (1.0 = no smoothing) |
| `max_gap` | 0.5 s | pose gap after which velocity history is dropped |
| `fall_speed` | 1.5 torso/s | downward hip speed that opens POSSIBLE_FALL |
| `angle_rate` | 120 °/s | torso rotation that opens POSSIBLE_FALL |
| `horizontal_angle` | 55° | lying band is [55°, 125°] for torso or leg angle |
| `upright_angle` | 30° | torso angle counted as upright |
| `hip_low_ratio` | 0.6 torso | hip height at or below this = hips near the floor |
| `possible_window` | 1.5 s | time allowed between the fast movement and lying |
| `alert_hold` | 1.0 s | lying time before FALL_ALERT |
| `recovery_hold` | 1.0 s | upright time before leaving FALL_ALERT |
| `recovery_cooldown` | 2.0 s | time in RECOVERY before NORMAL |
| `still_motion` | 0.3 torso/s | body motion below this counts as motionless |
| `still_alert` | 5.0 s | motionless time during an alert before escalating |

Changes made during tuning (on the `tune` split only): `horizontal_angle` 60 → 55; the leg-angle signal was added
(fall-03 falls toward the camera); knees are used for hip height when ankles are hidden, and the lying band now has
an upper bound (adl-30, dropping onto all fours near the camera).

## Test data and results

Public **UR Fall Detection Dataset** (B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps
and wireless accelerometer*, Computer Methods and Programs in Biomedicine, 2014), https://fenix.ur.edu.pl/~mkepski/ds/uf.html,
cam0 RGB half. We tuned on the `tune` split and ran `holdout` **once** after tuning.
These are per-clip outcomes on a small set, **not a formal accuracy claim**.

| Brief test case | Clips | Outcome |
|---|---|---|
| Standing still / walking | first seconds of every clip | no alerts |
| Sitting down | adl-07, adl-08, adl-09, adl-25 | no alerts |
| Bending / crouching | adl-01, adl-02, adl-04, adl-05, adl-28 | no alerts |
| Lying down intentionally | adl-10, adl-11, adl-36, adl-37 | **adl-10 and adl-11 alert** (sit on the bed edge, swing the legs up, lie down); adl-36 and adl-37 don't |
| Simulated fall + horizontal after rapid movement | fall-01 … fall-15 | 11/15 alert |
| Losing landmarks | adl-19 (dark room, pose found in 44% of frames) + unit tests | no alert, state frozen while lost |
| On all fours on the floor | adl-30, adl-31 | no alerts |

**Holdout: 16/21 clips as expected (non-falls 10/11, falls 6/10).** Tune: 9/10.

Why the holdout falls were missed (diagnosed after the run, not tuned on). All four reached POSSIBLE_FALL:
- fall-14 and fall-15: the clips are 2.0 s and 2.4 s long and end before `alert_hold` (1 s of lying) is reached.
- fall-06: the person lies flat (torso 91°), but hip height is 0.70, just over `hip_low_ratio`.
- fall-13: MediaPipe loses the person after the fall. The last poses it reports are upright and wrong.

### Holdout

| clip | expected | alerted | first alert (s) | pose found | result |
|---|---|---|---|---|---|
| adl-02 | none | no | - | 73% | OK |
| adl-05 | none | no | - | 94% | OK |
| adl-08 | none | no | - | 83% | OK |
| adl-09 | none | no | - | 100% | OK |
| adl-11 | none | yes | 9.0 | 92% | MISS |
| adl-19 | none | no | - | 44% | OK |
| adl-25 | none | no | - | 100% | OK |
| adl-28 | none | no | - | 99% | OK |
| adl-31 | none | no | - | 94% | OK |
| adl-36 | none | no | - | 83% | OK |
| adl-37 | none | no | - | 85% | OK |
| fall-06 | alert | no | - | 100% | MISS |
| fall-07 | alert | yes | 4.7 | 96% | OK |
| fall-08 | alert | yes | 2.8 | 100% | OK |
| fall-09 | alert | yes | 5.7 | 100% | OK |
| fall-10 | alert | yes | 3.0 | 99% | OK |
| fall-11 | alert | yes | 3.4 | 100% | OK |
| fall-12 | alert | yes | 2.9 | 100% | OK |
| fall-13 | alert | no | - | 95% | MISS |
| fall-14 | alert | no | - | 100% | MISS |
| fall-15 | alert | no | - | 87% | MISS |

### Tune

| clip | expected | alerted | first alert (s) | pose found | result |
|---|---|---|---|---|---|
| adl-01 | none | no | - | 82% | OK |
| adl-04 | none | no | - | 97% | OK |
| adl-07 | none | no | - | 93% | OK |
| adl-10 | none | yes | 8.9 | 97% | MISS |
| adl-30 | none | no | - | 55% | OK |
| fall-01 | alert | yes | 4.8 | 76% | OK |
| fall-02 | alert | yes | 3.0 | 100% | OK |
| fall-03 | alert | yes | 7.1 | 90% | OK |
| fall-04 | alert | yes | 2.1 | 99% | OK |
| fall-05 | alert | yes | 4.4 | 100% | OK |

## Limitations

- Rule-based baseline, **not medically validated**. Getting onto a bed and lying down with a quick swing of the torso can alert (adl-10, adl-11).
- One person only (`num_poses=1`). With several people the "first" pose may switch between them.
- Best with a roughly side-on camera. The leg-angle signal helps with falls toward the camera, but needs the ankles in view.
- `alert_hold` is 1 s because the dataset clips end soon after the fall, and very short clips still end too early. A real deployment would hold longer, which also cuts false alarms.
- Low light or the person leaving the frame causes pose loss. The state is frozen, not cleared, so an alert persists, but a fall in the dark may be missed. MediaPipe sometimes places a spurious skeleton on furniture (seen in fall-01).
- A head-down torso (beyond 125°) isn't counted as lying, so a fall that ends inverted, for example down stairs, may be missed.

## Troubleshooting

- `Activate.ps1 cannot be loaded`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- `Model not found`: run `python download_data.py`
- cv2 import errors or two OpenCV builds: use the venv. Don't install `opencv-python` next to mediapipe's `opencv-contrib-python`.
- UR clips look like two images side by side: add `--crop-right`.
