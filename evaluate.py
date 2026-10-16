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
    paths = []
    for pat in patterns:
        found = sorted(glob.glob(pat))
        if not found:
            print(f"warning: no clips match {pat}", file=sys.stderr)
        paths += found
    return paths


def score_clip(name, history):
    """One results-table row for a clip: fall-* clips should alert, adl-* clips should not."""
    alerts = [t for t, state, _ in history if state == FALL_ALERT]
    expected = name.startswith("fall")
    pose_pct = 100 * sum(p for _, _, p in history) / max(len(history), 1)
    return {"clip": name, "expected": "alert" if expected else "none", "alerted": "yes" if alerts else "no",
            "first alert (s)": f"{alerts[0]:.1f}" if alerts else "-", "pose found": f"{pose_pct:.0f}%",
            "result": "OK" if bool(alerts) == expected else "MISS"}


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
        if not history:
            print(f"| {name} | - | - | - | - | ERROR: no frames read |")
            continue
        row = score_clip(name, history)
        correct += row["result"] == "OK"
        print("| " + " | ".join(row.values()) + " |")
    print(f"\n{correct}/{len(paths)} clips as expected")
