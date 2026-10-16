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
