# Real-Time Fall Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python/OpenCV app that runs MediaPipe Pose Landmarker on a webcam or video, extracts torso angle and hip-motion features, and raises a potential-fall alert through a rule-based state machine. Thresholds are tuned and evaluated on the public UR Fall Detection dataset.

**Architecture:** There are two code modules. `fall_detector.py` is pure Python with no cv2 or mediapipe imports. It turns landmark points plus timestamps into a state and features, and it is fully unit-tested with synthetic poses. `main.py` is the glue: video input, MediaPipe, overlay and CLI. `evaluate.py` runs `main.run()` headless over the downloaded clips and prints a markdown results table. `download_data.py` fetches the model and clips with the stdlib only.

**Tech Stack:** Python 3.10.11 (installed), `mediapipe==0.10.35` (Tasks API; it brings in `opencv-contrib-python` and `numpy`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-fall-detection-brief.md`

## Global Constraints

- "Do not train or fine-tune any models." Use only the official pretrained `pose_landmarker_*.task`.
- "Complete a functional prototype within approximately two hours."
- "Avoid unnecessary dependencies and complicated architectures." The only dependencies are mediapipe and pytest. NumPy comes with mediapipe and isn't imported directly.
- "Use configurable thresholds rather than hardcoded values scattered throughout the code." Every threshold lives in `fall_detector.Config`.
- "Normalize coordinates before calculating velocities and distances." We divide by torso length (see Design decisions).
- "Use timestamps or actual frame intervals when calculating movement speed."
- "Handle missing or low-confidence landmarks safely."
- "Do not assume a particular operating system." The team uses Windows and PowerShell, so: no shell globs in instructions, no `&&`, and ASCII-only console output.
- Test data must be **public**. We use the UR Fall Detection Dataset (cam0 RGB). No self-recorded clips are used for the results.
- "If a labeled dataset is unavailable, report qualitative results rather than claiming formal model accuracy." UR Fall gives clip-level labels only, so report per-clip outcomes, not accuracy claims.

## Review Focus

1. **PowerShell does not expand `*.mp4`.** `evaluate.py "test_videos/tune/*.mp4"` must expand the glob itself, or it silently processes nothing. Pinned by `test_evaluate_expands_globs` (Task 4).
2. **The pose reappears at a different place after a dropout** (occlusion, walking out of frame). This must not create a fake velocity spike or a fall. Pinned by `test_reappearing_elsewhere_after_gap_is_not_a_fall` (Task 2).
3. **A low-visibility or degenerate pose** (all landmarks at one point, torso length 0) must not crash with ZeroDivisionError and is treated as "no pose". Pinned by `test_degenerate_torso_does_not_crash` and `test_low_visibility_is_treated_as_missing` (Task 2).
4. **A bad `--source` path, or a video whose FPS metadata reads 0**, must give a clear exit message rather than a cv2 traceback, and the fps falls back to 30. Pinned by `test_missing_source_exits_cleanly` (Task 3).
5. **Starting at t = 0.0.** `0.0` is falsy, so `x or t` style timer code breaks on the first frame. Every synthetic test starts at t=0, so it is pinned by the Task 2 suite. Also, `detect_for_video` requires strictly increasing int ms, which `run()` enforces (Task 3).

## Design decisions (agreed in discussion — don't re-litigate)

| Decision | Why |
|---|---|
| No `pose_estimator.py`. MediaPipe lives in `main.py` | About 20 lines of wrapper. A separate module adds an import boundary with no testability gain. |
| Pixel coordinates in, divided by **reference torso length** | Normalized 0–1 coordinates still depend on camera distance. Torso-lengths per second does not. The reference is updated only while upright, because lying toward the camera foreshortens the torso. |
| Extra feature: **hip height** = (ankle_y − hip_y) / torso | The brief's features can't tell **bending** (torso horizontal, hips high) from lying (hips near the floor). |
| A fast drop (speed **or** rotation) must happen **before** lying counts | Lying down slowly never enters POSSIBLE_FALL, so no alert. |
| Missing pose **freezes** the state and resets velocity history | A fallen person hidden by furniture keeps the alert, and the first frame after a gap can't produce a fake speed spike. |
| Ankles not visible means hip-height is ignored (angle alone decides) | Missing a fall is worse than a false alarm. Pinned by a test. |
| `alert_hold` defaults to **1.0 s**, not 2 s | UR fall clips last about 5 s with only about 1–2 s lying at the end. Documented in the README. |
| `--crop-right` flag | UR Fall `cam0` mp4s are 640×240 **depth \| RGB side by side**. The RGB half is the right 320×240. |
| Lite model by default, `--model` flag to try full | Real time on CPU first. |
| Pin `mediapipe==0.10.35`, not 1.x | The 0.10 Tasks API is known; the 1.x API is unverified. A dry-run install on this machine succeeds for both. |
| Use a venv | mediapipe pulls in `opencv-contrib-python`. The global env already has `opencv-python` 5.0, and two `cv2` packages can clash. |

## Interface contract (agree on this at 0:00 so both people can work in parallel)

```python
# fall_detector.py  (Person B)
NORMAL, POSSIBLE_FALL, FALL_ALERT, RECOVERY = "NORMAL", "POSSIBLE_FALL", "FALL_ALERT", "RECOVERY"
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 23, 24, 27, 28
@dataclass class Config: ...            # all thresholds
@dataclass class Features: angle: float; hip_speed: float; angle_rate: float; hip_height: float | None
class FallDetector:
    def __init__(self, config: Config | None = None): ...
    def update(self, points, t: float) -> tuple[str, Features | None]: ...
    # points: indexable by landmark id (list of 33 or dict) -> (x_px, y_px, visibility), or None

# main.py  (Person A)
DEFAULT_MODEL: str
def run(source: str, model: str = DEFAULT_MODEL, crop_right: bool = False, show: bool = True) -> list[tuple[float, str, bool]]
    # returns [(t_seconds, state, has_valid_pose)] per frame
```

## Team split and timeline (2 people)

| Time | Person A — pipeline | Person B — logic |
|---|---|---|
| 0:00–0:05 | Both: read this plan and agree on the interface contract | |
| 0:05–0:20 | **Task 1** setup, venv, download model + clips | **Task 2** starts (needs only pytest, which is already installed globally) |
| 0:20–0:55 | **Task 3** `main.py` (video, pose, overlay) | **Task 2** `fall_detector.py` + synthetic tests |
| 0:55–1:10 | **Task 4** `evaluate.py` | Review A's `main.py`; watch clips with the overlay |
| 1:10–1:40 | **Task 5** tuning together: A drives `evaluate.py`, B edits `Config` | |
| 1:40–2:00 | **Task 6** README: A writes setup/usage, B writes algorithm/thresholds/results | |

**Fallback if MediaPipe blocks A for more than 15 min:** try `pip install mediapipe==1.1.1` in the same venv and check the import line in Task 1 Step 3. If `mediapipe.tasks.python.vision` is missing, stay on 0.10.35 and ask for help. Don't switch to the legacy `mp.solutions.pose`, because the brief requires the Pose Landmarker task model.

**Git:** both work on `main` and own different files (A: `main.py`, `download_data.py`, `evaluate.py`, `requirements.txt`, `.gitignore`, `test_main.py`, `test_evaluate.py`; B: `fall_detector.py`, `test_fall_detector.py`). Run `git pull --rebase` before every push. README comes last.

## File map

| File | Owner | Responsibility |
|---|---|---|
| `requirements.txt` | A | `mediapipe==0.10.35`, `pytest` |
| `.gitignore` | A | venv, models, videos, caches |
| `download_data.py` | A | Download model(s) and UR Fall clips into `models/` and `test_videos/{tune,holdout}/` |
| `fall_detector.py` | B | Config, Features, FallDetector (features, smoothing, state machine) |
| `test_fall_detector.py` | B | Synthetic-pose tests for every brief test case that synthetic data can express |
| `main.py` | A | `frames()`, `make_landmarker()`, `to_points()`, `draw()`, `run()`, CLI |
| `test_main.py` | A | Frame reading, crop, timestamps, bad source, real pose found on a UR clip |
| `evaluate.py` | A | Headless run over clips, markdown table |
| `test_evaluate.py` | A | Glob expansion |
| `README.md` | both | Setup, model download, usage, algorithm, thresholds, limitations, results |

## Public test data: UR Fall Detection Dataset

Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html. Files are `https://fenix.ur.edu.pl/~mkepski/ds/data/{name}-cam0.mp4`, 640×240 at 30 fps, depth on the left and RGB on the right. Citation: B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014.

ADL (non-fall) clips were labeled by viewing them on 2026-10-16:

| Brief test case | Clips |
|---|---|
| 1 Standing still, 2 Walking | The first 1–2 s of every clip (the person walks in and stands) |
| 3 Sitting down | adl-07, adl-08 (armchair, side), adl-09 (bed edge), adl-25 (chair, frontal) |
| 4 Bending / picking up | adl-04, adl-05, adl-28 (bend); adl-01, adl-02 (crouch) |
| 5 Lying down intentionally | adl-10, adl-11 (lie on bed), adl-36, adl-37 (lie on bed, different room) |
| 6 Simulated fall, 7 Horizontal after rapid movement | fall-01 … fall-15 |
| 8 Losing reliable landmarks | adl-19 (very dark room) + synthetic tests |
| Hard negative: on all fours on the floor | adl-30, adl-31 |

Split: **tune** = fall-01..05, adl-01, adl-04, adl-07, adl-10, adl-30. **holdout** = fall-06..15, adl-02, adl-05, adl-08, adl-09, adl-11, adl-19, adl-25, adl-28, adl-31, adl-36, adl-37. Tune only on `tune`. Run `holdout` **once** at the end and report that.

---

### Task 1: Environment, dependencies, data download (Person A)

**Files:**
- Create: `requirements.txt`, `.gitignore`, `download_data.py`

**Interfaces:**
- Produces: `models/pose_landmarker_lite.task`, `models/pose_landmarker_full.task`, `test_videos/tune/*.mp4`, `test_videos/holdout/*.mp4` (named `{clip}-cam0.mp4`)

- [ ] **Step 1: Create `requirements.txt`**

```text
# mediapipe pulls in opencv-contrib-python and numpy. Don't add opencv-python: two cv2 packages clash.
mediapipe==0.10.35
pytest
```

- [ ] **Step 2: Create `.gitignore`**

```text
.venv/
__pycache__/
.pytest_cache/
models/
test_videos/
```

- [ ] **Step 3: Create the venv, install, and check the Tasks API import (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import mediapipe as mp, cv2; from mediapipe.tasks.python import vision; print(mp.__version__, cv2.__version__, vision.PoseLandmarker)"
```

Expected: `0.10.35 5.0.0 <class '...PoseLandmarker'>` (the cv2 version may differ). If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

- [ ] **Step 4: Create `download_data.py`**

```python
"""Download the MediaPipe Pose Landmarker models and public UR Fall Detection clips.

Dataset: https://fenix.ur.edu.pl/~mkepski/ds/uf.html (Kwolek & Kepski, 2014).
cam0 videos are 640x240: depth on the left, RGB on the right (use --crop-right).
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task"
CLIP_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data/{}-cam0.mp4"
CLIPS = {
    "tune": ["fall-01", "fall-02", "fall-03", "fall-04", "fall-05",
             "adl-01", "adl-04", "adl-07", "adl-10", "adl-30"],
    "holdout": [f"fall-{i:02d}" for i in range(6, 16)] +
               ["adl-02", "adl-05", "adl-08", "adl-09", "adl-11", "adl-19",
                "adl-25", "adl-28", "adl-31", "adl-36", "adl-37"],
}


def fetch(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    tmp = dest.with_name(dest.name + ".part")  # a killed download never looks complete
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


if __name__ == "__main__":
    for size in ("lite", "full"):
        fetch(MODEL_URL.format(size), ROOT / "models" / f"pose_landmarker_{size}.task")
    for split, names in CLIPS.items():
        for name in names:
            fetch(CLIP_URL.format(name), ROOT / "test_videos" / split / f"{name}-cam0.mp4")
    print("done")
```

- [ ] **Step 5: Run it and check the counts**

```powershell
python download_data.py
(Get-ChildItem test_videos\tune).Count; (Get-ChildItem test_videos\holdout).Count; Get-ChildItem models
```

Expected: `10`, `21`, and two `.task` files (lite about 5 MB, full about 9 MB). Re-running prints only `done`.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt .gitignore download_data.py
git commit -m "chore: deps, gitignore, model and UR Fall clip downloader"
```

---

### Task 2: `fall_detector.py` — features + state machine, TDD (Person B)

**Files:**
- Create: `fall_detector.py`
- Test: `test_fall_detector.py`

**Interfaces:**
- Consumes: nothing (pure Python)
- Produces: exactly the `fall_detector.py` block in the Interface contract above

- [ ] **Step 1: Write the failing tests**

```python
import math

from fall_detector import (FALL_ALERT, L_ANK, L_HIP, L_SH, NORMAL, POSSIBLE_FALL, R_ANK, R_HIP,
                           R_SH, RECOVERY, FallDetector)

FPS = 30
STAND = dict(angle=0.0, hip_y=300.0, ankle_y=460.0)
LYING = dict(angle=85.0, hip_y=420.0, ankle_y=420.0)


def pose(angle=0.0, hip_y=300.0, ankle_y=460.0, torso=100.0, vis=0.9, ankle_vis=None):
    """Synthetic landmarks in pixels: torso `torso` px long, tilted `angle` deg from vertical."""
    a = math.radians(angle)
    sx, sy = 200 + torso * math.sin(a), hip_y - torso * math.cos(a)
    av = vis if ankle_vis is None else ankle_vis
    pts = {i: (200.0, hip_y, vis) for i in (L_HIP, R_HIP)}
    pts.update({i: (sx, sy, vis) for i in (L_SH, R_SH)})
    pts.update({i: (200.0, ankle_y, av) for i in (L_ANK, R_ANK)})
    return pts


def ramp(n, start, end, **extra):
    """n poses linearly interpolated from keyword dict `start` to `end` (end included)."""
    return [pose(**{k: start[k] + (end[k] - start[k]) * (i + 1) / n for k in start}, **extra)
            for i in range(n)]


def run(frames, det=None):
    det = det or FallDetector()
    return [det.update(p, i / FPS)[0] for i, p in enumerate(frames)]


# 1 s standing, 0.4 s fast fall, 2 s lying
FALL = [pose(**STAND)] * 30 + ramp(12, STAND, LYING) + [pose(**LYING)] * 60


def test_features_report_angle_from_vertical():
    _, f = FallDetector().update(pose(angle=0.0), 0.0)
    assert abs(f.angle) < 1
    _, f = FallDetector().update(pose(angle=85.0), 0.0)
    assert abs(f.angle - 85) < 1


def test_standing_still_stays_normal():
    assert set(run([pose(**STAND)] * 90)) == {NORMAL}


def test_fast_fall_then_lying_raises_alert_in_order():
    states = run(FALL)
    first = states.index(POSSIBLE_FALL)
    assert set(states[:first]) == {NORMAL}
    assert first < states.index(FALL_ALERT)
    assert states[-1] == FALL_ALERT


def test_slow_lie_down_never_alerts():
    states = run([pose(**STAND)] * 30 + ramp(120, STAND, LYING) + [pose(**LYING)] * 90)
    assert POSSIBLE_FALL not in states and FALL_ALERT not in states


def test_sitting_down_fast_times_out_to_normal():
    sit = dict(angle=5.0, hip_y=380.0, ankle_y=460.0)
    states = run([pose(**STAND)] * 30 + ramp(9, STAND, sit) + [pose(**sit)] * 90)
    assert POSSIBLE_FALL in states and FALL_ALERT not in states
    assert states[-1] == NORMAL


def test_bending_is_not_lying():
    bend = dict(angle=80.0, hip_y=300.0, ankle_y=460.0)
    states = run([pose(**STAND)] * 30 + ramp(12, STAND, bend) + [pose(**bend)] * 90)
    assert FALL_ALERT not in states
    assert states[-1] == NORMAL


def test_fall_with_hidden_ankles_still_alerts():
    frames = [pose(**STAND)] * 30 + ramp(12, STAND, LYING, ankle_vis=0.1) + [pose(**LYING, ankle_vis=0.1)] * 60
    assert run(frames)[-1] == FALL_ALERT


def test_pose_loss_freezes_alert():
    states = run(FALL + [None] * 60 + [pose(**LYING)] * 30)
    assert states[len(FALL) - 1:] == [FALL_ALERT] * 91


def test_reappearing_elsewhere_after_gap_is_not_a_fall():
    moved = dict(angle=0.0, hip_y=450.0, ankle_y=610.0)
    states = run([pose(**STAND)] * 30 + [None] * 18 + [pose(**moved)] * 30)
    assert set(states) == {NORMAL}


def test_standing_up_after_alert_recovers_then_normal():
    states = run(FALL + ramp(12, LYING, STAND) + [pose(**STAND)] * 120)
    assert states.index(FALL_ALERT) < states.index(RECOVERY)
    assert states[-1] == NORMAL


def test_low_visibility_is_treated_as_missing():
    state, f = FallDetector().update(pose(vis=0.1), 0.0)
    assert state == NORMAL and f is None


def test_degenerate_torso_does_not_crash():
    state, f = FallDetector().update(pose(torso=0.0), 0.0)
    assert state == NORMAL and f is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest test_fall_detector.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'fall_detector'`

- [ ] **Step 3: Implement `fall_detector.py`**

```python
"""Rule-based fall detection from pose landmarks.

Pure Python on purpose (no OpenCV/MediaPipe) so it can be unit-tested with synthetic poses.
Input: landmark points in pixels plus a timestamp. Output: state and the features behind it.
"""
import math
from dataclasses import dataclass

# MediaPipe Pose landmark indices
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 23, 24, 27, 28

NORMAL, POSSIBLE_FALL, FALL_ALERT, RECOVERY = "NORMAL", "POSSIBLE_FALL", "FALL_ALERT", "RECOVERY"


@dataclass
class Config:
    min_visibility: float = 0.5     # landmark pairs below this count as missing
    smooth_alpha: float = 0.5       # EMA weight of the newest frame (1.0 = no smoothing)
    max_gap: float = 0.5            # s without a valid pose before velocity history is dropped
    fall_speed: float = 1.5         # downward hip speed (torso-lengths/s) that opens POSSIBLE_FALL
    angle_rate: float = 120.0       # torso rotation (deg/s) that opens POSSIBLE_FALL
    horizontal_angle: float = 60.0  # torso angle from vertical counted as lying
    upright_angle: float = 30.0     # torso angle from vertical counted as upright
    hip_low_ratio: float = 0.6      # hip height (torso-lengths above ankles) below this = near floor
    possible_window: float = 1.5    # s allowed between the fast movement and lying
    alert_hold: float = 1.0         # s lying after a fast movement before FALL_ALERT
    recovery_hold: float = 1.0      # s upright before leaving FALL_ALERT
    recovery_cooldown: float = 2.0  # s in RECOVERY before NORMAL


@dataclass
class Features:
    angle: float              # torso angle from vertical, degrees: 0 upright, 90 horizontal
    hip_speed: float          # downward hip speed, torso-lengths/s (negative = moving up)
    angle_rate: float         # |change in angle|, degrees/s
    hip_height: float | None  # (ankle_y - hip_y) / torso; None when ankles are not visible


def _mid(points, a, b, min_vis):
    """Midpoint of landmarks a and b, or None if neither is visible enough."""
    pa, pb = points[a], points[b]
    if max(pa[2], pb[2]) < min_vis:
        return None
    return (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2


class FallDetector:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.state = NORMAL
        self.state_since = 0.0
        self.lying_since = None
        self.upright_since = None
        self.ref_torso = None  # torso length (px) while upright; scales speeds and heights
        self._ema = None       # smoothed [hip_x, hip_y, sh_x, sh_y, ankle_y | None]
        self._prev = None      # (t, hip_y, angle) of the last valid frame

    def update(self, points, t):
        """points: indexable by landmark id -> (x_px, y_px, visibility), or None when no pose.
        t: timestamp in seconds. Returns (state, Features | None). A missing pose freezes the state."""
        f = self._features(points, t)
        if f is not None:
            self._step(f, t)
        return self.state, f

    def _features(self, points, t):
        c = self.cfg
        if points is None:
            return None
        hip = _mid(points, L_HIP, R_HIP, c.min_visibility)
        sh = _mid(points, L_SH, R_SH, c.min_visibility)
        if hip is None or sh is None:
            return None
        ank = _mid(points, L_ANK, R_ANK, c.min_visibility)
        raw = [hip[0], hip[1], sh[0], sh[1], ank[1] if ank else None]

        # after a gap, restart smoothing and velocity so a jump in position is not a "fall"
        gap = self._prev is None or t - self._prev[0] > c.max_gap
        if gap:
            self._ema = raw
        else:
            a = c.smooth_alpha
            self._ema = [r if r is None or e is None else a * r + (1 - a) * e
                         for r, e in zip(raw, self._ema)]
        hx, hy, sx, sy, ay = self._ema

        torso = math.hypot(sx - hx, sy - hy)
        if torso < 1.0:
            return None
        angle = math.degrees(math.atan2(abs(sx - hx), hy - sy))
        if self.ref_torso is None:
            self.ref_torso = torso
        elif angle <= c.upright_angle:
            self.ref_torso = 0.9 * self.ref_torso + 0.1 * torso

        if gap:
            hip_speed = angle_rate = 0.0
        else:
            dt = max(t - self._prev[0], 1e-3)
            hip_speed = (hy - self._prev[1]) / dt / self.ref_torso
            angle_rate = abs(angle - self._prev[2]) / dt
        self._prev = (t, hy, angle)
        hip_height = None if ay is None else (ay - hy) / self.ref_torso
        return Features(angle, hip_speed, angle_rate, hip_height)

    def _enter(self, state, t):
        self.state, self.state_since = state, t

    def _step(self, f, t):
        c = self.cfg
        lying = f.angle >= c.horizontal_angle and (f.hip_height is None or f.hip_height <= c.hip_low_ratio)
        upright = f.angle <= c.upright_angle
        # explicit None checks: t == 0.0 is a valid start time
        if not lying:
            self.lying_since = None
        elif self.lying_since is None:
            self.lying_since = t
        if not upright:
            self.upright_since = None
        elif self.upright_since is None:
            self.upright_since = t

        if self.state == NORMAL:
            if f.hip_speed >= c.fall_speed or f.angle_rate >= c.angle_rate:
                self._enter(POSSIBLE_FALL, t)
        elif self.state == POSSIBLE_FALL:
            if self.lying_since is not None and t - self.lying_since >= c.alert_hold:
                self._enter(FALL_ALERT, t)
            elif self.lying_since is None and t - self.state_since > c.possible_window:
                self._enter(NORMAL, t)
        elif self.state == FALL_ALERT:
            if self.upright_since is not None and t - self.upright_since >= c.recovery_hold:
                self._enter(RECOVERY, t)
        elif t - self.state_since >= c.recovery_cooldown:  # RECOVERY
            self._enter(NORMAL, t)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest test_fall_detector.py -q`
Expected: `12 passed`. If one fails, fix the detector, not the test. The scenarios encode the brief.

- [ ] **Step 5: Commit**

```powershell
git add fall_detector.py test_fall_detector.py
git commit -m "feat: rule-based fall detector with synthetic-pose tests"
```

---

### Task 3: `main.py` — video input, MediaPipe, overlay, CLI (Person A)

**Files:**
- Create: `main.py`
- Test: `test_main.py`

**Interfaces:**
- Consumes: `FallDetector`, `FALL_ALERT`, `POSSIBLE_FALL`, `RECOVERY`, `NORMAL` from `fall_detector.py`. Until Task 2 lands, A can stub `fall_detector.py` with the contract signatures returning `(NORMAL, None)`. Don't commit the stub.
- Produces: `DEFAULT_MODEL: str`, `frames(source: str, crop_right: bool) -> Iterator[tuple[float, ndarray]]`, `run(source, model=DEFAULT_MODEL, crop_right=False, show=True) -> list[tuple[float, str, bool]]`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from main import DEFAULT_MODEL, frames, run

CLIP = Path(__file__).parent / "test_videos" / "tune" / "fall-01-cam0.mp4"
need_clip = pytest.mark.skipif(not CLIP.exists(), reason="run download_data.py first")


@need_clip
def test_frames_crop_right_and_increasing_timestamps():
    out = list(frames(str(CLIP), crop_right=True))
    assert len(out) > 100
    assert out[0][1].shape[:2] == (240, 320)
    assert all(b[0] > a[0] for a, b in zip(out, out[1:]))


def test_missing_source_exits_cleanly():
    with pytest.raises(SystemExit):
        list(frames("no_such_file.mp4", crop_right=False))


@need_clip
def test_pipeline_finds_the_person_on_a_ur_clip():
    history = run(str(CLIP), DEFAULT_MODEL, crop_right=True, show=False)
    assert len(history) > 100
    assert sum(has_pose for _, _, has_pose in history) > 0.7 * len(history)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest test_main.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement `main.py`**

```python
"""Real-time fall detection: webcam/video -> MediaPipe Pose Landmarker -> FallDetector -> overlay.

Usage:
  python main.py                                   # webcam 0
  python main.py --source path/to/video.mp4
  python main.py --source test_videos/tune/fall-01-cam0.mp4 --crop-right   # UR Fall clips
Keys: q or Esc = quit, space = pause/resume.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from fall_detector import FALL_ALERT, NORMAL, POSSIBLE_FALL, RECOVERY, FallDetector

DEFAULT_MODEL = str(Path(__file__).parent / "models" / "pose_landmarker_lite.task")
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
COLORS = {NORMAL: (0, 180, 0), POSSIBLE_FALL: (0, 200, 255), FALL_ALERT: (0, 0, 255), RECOVERY: (255, 180, 0)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_landmarker(model_path):
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path} (run: python download_data.py)")
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def frames(source, crop_right):
    """Yield (t_seconds, bgr_frame). Files use frame index / fps so runs are reproducible."""
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.monotonic(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if crop_right:
                frame = frame[:, frame.shape[1] // 2:]
            yield (time.monotonic() - start if is_cam else i / fps), frame
            i += 1
    finally:
        cap.release()


def to_points(result, w, h):
    """First detected pose as 33 x (x_px, y_px, visibility), or None."""
    if not result.pose_landmarks:
        return None
    return [(lm.x * w, lm.y * h, lm.visibility) for lm in result.pose_landmarks[0]]


def draw(frame, points, state, feats):
    h, w = frame.shape[:2]
    if points:
        for a, b in BONES:
            if min(points[a][2], points[b][2]) > 0.5:
                cv2.line(frame, (int(points[a][0]), int(points[a][1])),
                         (int(points[b][0]), int(points[b][1])), (255, 255, 255), 2)
    cv2.rectangle(frame, (0, 0), (w, 30), COLORS[state], -1)
    cv2.putText(frame, state if feats else f"{state}  (NO POSE)", (8, 21), FONT, 0.6, (0, 0, 0), 2)
    if feats:
        hh = "-" if feats.hip_height is None else f"{feats.hip_height:.2f}"
        cv2.putText(frame, f"angle {feats.angle:3.0f}  hip v {feats.hip_speed:+.2f}  hip h {hh}",
                    (8, h - 32), FONT, 0.5, (255, 255, 255), 1)
    if state == FALL_ALERT and int(time.monotonic() * 4) % 2 == 0:  # flashing
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
        cv2.putText(frame, "POTENTIAL FALL!", (w // 2 - 120, h // 2), FONT, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, "q/Esc: quit   space: pause", (8, h - 10), FONT, 0.45, (200, 200, 200), 1)


def run(source, model=DEFAULT_MODEL, crop_right=False, show=True):
    """Process one source. Returns [(t, state, has_valid_pose)] per frame."""
    detector, history, last_ms = FallDetector(), [], -1
    landmarker = make_landmarker(model)
    try:
        for t, frame in frames(source, crop_right):
            last_ms = max(int(t * 1000), last_ms + 1)  # VIDEO mode needs strictly increasing ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
            if frame.shape[1] < 640:  # UR clips are 320 px wide: upscale for a readable demo
                frame = cv2.resize(frame, None, fx=2, fy=2)
            points = to_points(result, frame.shape[1], frame.shape[0])
            state, feats = detector.update(points, t)
            history.append((t, state, feats is not None))
            if show:
                draw(frame, points, state, feats)
                cv2.imshow("Fall detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)
    finally:
        landmarker.close()
        if show:
            cv2.destroyAllWindows()
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index (e.g. 0) or video file path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pose_landmarker_{lite,full}.task path")
    p.add_argument("--crop-right", action="store_true", help="use right half of each frame (UR Fall: depth|RGB)")
    a = p.parse_args()
    run(a.source, a.model, a.crop_right)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest test_main.py -q`
Expected: `3 passed`. If `lm.visibility` is `None` (an API change), the pose test fails with a TypeError. Fix it in `to_points` with `lm.visibility or 0.0` and report it in the README troubleshooting section.

- [ ] **Step 5: Manual demo check (this is the deliverable)**

```powershell
python main.py --source test_videos\tune\fall-01-cam0.mp4 --crop-right
python main.py
```

Expected: a 640×480 window with a white skeleton, a state bar on top, and an angle/speed line at the bottom. On fall-01 the bar goes green → yellow → red with a flashing "POTENTIAL FALL!" (if it doesn't, that's for Task 5, not a Task 3 bug). The webcam opens; `q` quits.

- [ ] **Step 6: Commit**

```powershell
git add main.py test_main.py
git commit -m "feat: video/webcam pipeline with MediaPipe pose and OpenCV overlay"
```

---

### Task 4: `evaluate.py` — headless results table (Person A)

**Files:**
- Create: `evaluate.py`
- Test: `test_evaluate.py`

**Interfaces:**
- Consumes: `main.run`, `main.DEFAULT_MODEL`, `fall_detector.FALL_ALERT`
- Produces: `expand(patterns: list[str]) -> list[str]`; CLI that prints a markdown table

- [ ] **Step 1: Write the failing test**

```python
from evaluate import expand


def test_evaluate_expands_globs(tmp_path):
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4"):
        (tmp_path / name).write_bytes(b"")
    found = expand([str(tmp_path / "*.mp4")])
    assert [p.split("\\")[-1].split("/")[-1] for p in found] == ["adl-01-cam0.mp4", "fall-01-cam0.mp4"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest test_evaluate.py -q`
Expected: `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""Run the detector headless over clips and print a markdown results table.

Usage: python evaluate.py "test_videos/tune/*.mp4"   (quotes OK; globs are expanded here for PowerShell)
fall-* clips are expected to alert, adl-* clips are not. Output is ASCII-only for Windows consoles.
"""
import glob
import sys
from pathlib import Path

from fall_detector import FALL_ALERT
from main import DEFAULT_MODEL, run


def expand(patterns):
    return [p for pat in patterns for p in sorted(glob.glob(pat))]


if __name__ == "__main__":
    paths = expand(sys.argv[1:] or ["test_videos/*/*.mp4"])
    if not paths:
        raise SystemExit("No clips found (run: python download_data.py)")
    print("| clip | expected | alerted | first alert (s) | pose found | result |")
    print("|---|---|---|---|---|---|")
    correct = 0
    for path in paths:
        name = Path(path).stem.replace("-cam0", "")
        history = run(path, DEFAULT_MODEL, crop_right=True, show=False)
        alerts = [t for t, state, _ in history if state == FALL_ALERT]
        expected = name.startswith("fall")
        ok = bool(alerts) == expected
        correct += ok
        pose_pct = 100 * sum(p for _, _, p in history) / max(len(history), 1)
        print(f"| {name} | {'alert' if expected else 'none'} | {'yes' if alerts else 'no'} | "
              f"{f'{alerts[0]:.1f}' if alerts else '-'} | {pose_pct:.0f}% | {'OK' if ok else 'MISS'} |")
    print(f"\n{correct}/{len(paths)} clips as expected")
```

- [ ] **Step 4: Run the tests and a real evaluation**

```powershell
python -m pytest -q
python evaluate.py "test_videos/tune/*.mp4"
```

Expected: all tests pass (12 + 3 + 1 = `16 passed`), and a 10-row table plus `N/10 clips as expected`. Any N is fine here, because tuning is Task 5.

- [ ] **Step 5: Commit**

```powershell
git add evaluate.py test_evaluate.py
git commit -m "feat: headless evaluation table over UR Fall clips"
```

---

### Task 5: Threshold tuning on the `tune` split, then one holdout run (both, timebox 30 min)

**Files:**
- Modify: `fall_detector.py` (`Config` defaults only)

**Interfaces:**
- Consumes: `evaluate.py`, `main.py --crop-right`
- Produces: final `Config` defaults; `results_tune.md` and `results_holdout.md` tables (pasted into the README in Task 6)

- [ ] **Step 1: Baseline**

Run: `python evaluate.py "test_videos/tune/*.mp4" > results_tune.md` and read it.

- [ ] **Step 2: Diagnose each MISS by watching it**

Run: `python main.py --source test_videos\tune\<clip>-cam0.mp4 --crop-right`. Press space to pause at the event and read `angle`, `hip v` and `hip h` from the overlay. Change **one** `Config` field per iteration, using this table:

| Symptom (on overlay) | Field to change |
|---|---|
| Fall clip stays green (never yellow) | lower `fall_speed` (try 1.0) or `angle_rate` (try 90) |
| Goes yellow, back to green, person clearly on floor | `hip h` shows > `hip_low_ratio`? raise `hip_low_ratio` (try 0.8); `angle` < 60? lower `horizontal_angle` (try 50) |
| Goes yellow, clip ends before red | lower `alert_hold` (try 0.7) or raise `possible_window` (try 2.0) |
| ADL bend/crouch goes red | lower `hip_low_ratio` |
| ADL lie-on-bed (adl-10) goes red | expected limitation if the person flopped fast; note it, don't over-fit |
| `(NO POSE)` most of the clip | try `python main.py ... --model models\pose_landmarker_full.task`; if better, switch `DEFAULT_MODEL` |

- [ ] **Step 3: After each change, run the unit tests and the tune set**

```powershell
python -m pytest -q
python evaluate.py "test_videos/tune/*.mp4" > results_tune.md
```

Expected: tests still pass. If a Config change breaks a synthetic test, the change is too aggressive for that real scenario (sit, bend, slow lie). Back it off rather than editing the test.

- [ ] **Step 4: Stop at the timebox and run the holdout once**

Run: `python evaluate.py "test_videos/holdout/*.mp4" > results_holdout.md`
Don't tune after seeing it. Those are the numbers we report.

- [ ] **Step 5: Commit**

```powershell
git add fall_detector.py results_tune.md results_holdout.md
git commit -m "tune: Config thresholds on UR Fall tune split; add tune/holdout results"
```

---

### Task 6: README (both)

**Files:**
- Modify: `README.md` (currently UTF-16 with one heading. Overwrite it as UTF-8 and keep the heading text `K4-Track4-Capstone-TeamImback`.)

- [ ] **Step 1: Write `README.md`**

The content is below. Paste the actual tables from `results_tune.md` and `results_holdout.md`, and the final `Config` values from `fall_detector.py`, where marked with `<<paste ...>>`. Those values only exist after Task 5.

````markdown
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

## Run

```powershell
python main.py                                                        # webcam
python main.py --source my_video.mp4                                  # any video
python main.py --source test_videos\tune\fall-01-cam0.mp4 --crop-right # UR Fall clip (depth|RGB)
python main.py --model models\pose_landmarker_full.task               # more accurate, slower
python evaluate.py "test_videos/holdout/*.mp4"                        # results table
python -m pytest -q                                                   # unit tests
```

Keys: `q`/`Esc` quit, `space` pause.

## How it works

1. MediaPipe gives 33 landmarks per frame. We use shoulders, hips and ankles, ignoring a pair when neither point has visibility ≥ `min_visibility`.
2. Light exponential smoothing (`smooth_alpha`) on the mid-hip, mid-shoulder and ankle points.
3. Features, normalized by the person's **torso length** measured while upright (so camera distance doesn't matter):
   - **angle**: torso (mid-hip → mid-shoulder) angle from vertical, 0° upright, 90° horizontal
   - **hip v**: downward hip speed in torso-lengths/s, using real timestamps
   - **angle rate**: deg/s
   - **hip h**: hip height above the ankles in torso-lengths (separates *bending* from *lying*)
4. State machine:
   - NORMAL → POSSIBLE_FALL: hip v ≥ `fall_speed` **or** angle rate ≥ `angle_rate`
   - POSSIBLE_FALL → FALL_ALERT: lying (angle ≥ `horizontal_angle` and hip h ≤ `hip_low_ratio`) for `alert_hold` s
   - POSSIBLE_FALL → NORMAL: not lying within `possible_window` s (e.g. sitting down fast)
   - FALL_ALERT → RECOVERY: upright (angle ≤ `upright_angle`) for `recovery_hold` s
   - RECOVERY → NORMAL after `recovery_cooldown` s
   - No pose: the state is frozen and velocity history resets after `max_gap` s (no fake spikes when the person reappears)

Lying down **slowly** never passes the speed gate, so it doesn't alert.

## Thresholds (`fall_detector.Config`)

<<paste final Config values as a table: field | value | meaning>>

## Test data and results

Public **UR Fall Detection Dataset** (Kwolek & Kepski, *Computer Methods and Programs in Biomedicine*, 2014),
https://fenix.ur.edu.pl/~mkepski/ds/uf.html, cam0 RGB. Tuned on the `tune` split; `holdout` was run once after tuning.
Clip-level outcomes only. This is not a formal accuracy claim.

| Brief test case | Clips |
|---|---|
| Standing still / walking | first seconds of every clip |
| Sitting down | adl-07, adl-08, adl-09, adl-25 |
| Bending / crouching | adl-01, adl-02, adl-04, adl-05, adl-28 |
| Lying down intentionally | adl-10, adl-11, adl-36, adl-37 |
| Simulated fall + horizontal after rapid movement | fall-01 … fall-15 |
| Losing landmarks | adl-19 (dark room) + unit tests |
| On all fours on the floor | adl-30, adl-31 |

### Holdout
<<paste results_holdout.md>>

### Tune
<<paste results_tune.md>>

## Limitations

- Rule-based baseline, **not medically validated**. Quickly lying or flopping onto a bed or the floor can alert.
- One person only (`num_poses=1`). With several people the "first" pose may switch between them.
- Best with a roughly side-on camera. A fall straight toward the camera is foreshortened and the angle barely changes.
- `alert_hold` is about 1 s because the dataset clips end soon after the fall. A real deployment would hold longer.
- Low light causes pose loss (see adl-19). The state is frozen, not cleared, so an alert persists but a fall in the dark may be missed.

## Troubleshooting

- `Activate.ps1 cannot be loaded`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- `Model not found`: run `python download_data.py`
- cv2 import errors or two OpenCV builds: use the venv. Don't install `opencv-python` next to mediapipe's `opencv-contrib-python`.
- UR clips look like two images side by side: add `--crop-right`.
````

- [ ] **Step 2: Fill in the `<<paste>>` markers, then check that none are left**

Run: `Select-String -Path README.md -Pattern "<<paste"`
Expected: no output.

- [ ] **Step 3: Final check from a clean shell**

```powershell
python -m pytest -q
python main.py --source test_videos\holdout\fall-06-cam0.mp4 --crop-right
```

Expected: `16 passed`, and the demo window shows the skeleton and state bar.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: setup, algorithm, thresholds, results, limitations"
```

---

## Spec coverage check

| Brief requirement | Where |
|---|---|
| §3 MediaPipe Pose Landmarker, OpenCV, (NumPy via mediapipe) | Task 1, Task 3 |
| §4.1 webcam or video input | Task 3 `frames()`, `--source` |
| §4.3 visibility check | Task 2 `_mid` + `min_visibility`; test low-visibility |
| §4.4 angle, hip displacement, hip velocity, horizontal duration | Task 2 `Features`, `lying_since` |
| §5 four states, configurable thresholds | Task 2 `Config`, `_step` |
| §5 normalize, timestamps, smoothing, missing landmarks, temporal window | Task 2 (torso norm, `t`, EMA, freeze + gap reset, `alert_hold`/`possible_window`) |
| §6 skeleton, state, angle/hip indicators, distinct warning, start/stop instructions, CLI source switch | Task 3 `draw()`, argparse |
| §7 structure (simplified, allowed) | File map |
| §8 eight test cases recorded | Public-data table, Task 4, Task 5, README |
| §10 deliverables 1–7 | Tasks 1–6 |
