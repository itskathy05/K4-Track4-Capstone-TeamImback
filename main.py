"""Real-time fall detection: webcam/video -> MediaPipe Pose Landmarker -> FallDetector -> overlay.

Usage:
  python main.py                                   # webcam 0
  python main.py --source path/to/video.mp4
  python main.py --source test_videos/tune/fall-01-cam0.mp4 --crop-right   # UR Fall clips
Keys: q or Esc = quit, space = pause/resume.
"""
import argparse
import contextlib
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
    return [(lm.x * w, lm.y * h, lm.visibility or 0.0) for lm in result.pose_landmarks[0]]


def draw(frame, points, state, feats, key_hint=True):
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
    if key_hint:
        cv2.putText(frame, "q/Esc: quit   space: pause", (8, h - 10), FONT, 0.45, (200, 200, 200), 1)


def process(source, model=DEFAULT_MODEL, crop_right=False, detector=None):
    """Yield (t, frame, points, state, features) per frame. Shared by the CLI and the web demo."""
    detector, last_ms = detector or FallDetector(), -1
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
            yield t, frame, points, state, feats
    finally:
        landmarker.close()


def run(source, model=DEFAULT_MODEL, crop_right=False, show=True):
    """Process one source. Returns [(t, state, has_valid_pose)] per frame."""
    history = []
    try:
        with contextlib.closing(process(source, model, crop_right)) as stream:
            for t, frame, points, state, feats in stream:
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
