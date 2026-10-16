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
