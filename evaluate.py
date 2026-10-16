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
