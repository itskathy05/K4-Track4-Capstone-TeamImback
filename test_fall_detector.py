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
