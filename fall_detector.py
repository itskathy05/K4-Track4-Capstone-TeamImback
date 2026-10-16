"""Rule-based fall detection from pose landmarks.

Pure Python on purpose (no OpenCV/MediaPipe) so it can be unit-tested with synthetic poses.
Input: landmark points in pixels plus a timestamp. Output: state and the features behind it.
"""
import math
from dataclasses import dataclass

# MediaPipe Pose landmark indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 23, 24, 25, 26, 27, 28

NORMAL, POSSIBLE_FALL, FALL_ALERT, RECOVERY = "NORMAL", "POSSIBLE_FALL", "FALL_ALERT", "RECOVERY"


@dataclass
class Config:
    min_visibility: float = 0.5     # landmark pairs below this count as missing
    smooth_alpha: float = 0.5       # EMA weight of the newest frame (1.0 = no smoothing)
    max_gap: float = 0.5            # s without a valid pose before velocity history is dropped
    fall_speed: float = 1.5         # downward hip speed (torso-lengths/s) that opens POSSIBLE_FALL
    angle_rate: float = 120.0       # torso rotation (deg/s) that opens POSSIBLE_FALL
    horizontal_angle: float = 55.0  # lying = torso (or shoulder->ankle) within [this, 180 - this] deg;
                                    # beyond 180 - this the torso is head-down (crawling), not flat
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
    hip_height: float | None  # (ankle_y - hip_y) / torso, knees if ankles hidden; None if neither visible
    leg_angle: float | None   # shoulder->ankle line angle from vertical (catches torsos foreshortened
                              # by falling toward the camera); None when ankles are not visible


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
        self._ema = None       # smoothed [hip_x, hip_y, sh_x, sh_y, ankle_x, ankle_y, knee_y] (None if hidden)
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
        ank = _mid(points, L_ANK, R_ANK, c.min_visibility) or (None, None)
        knee = _mid(points, L_KNEE, R_KNEE, c.min_visibility) or (None, None)
        raw = [hip[0], hip[1], sh[0], sh[1], ank[0], ank[1], knee[1]]

        # after a gap, restart smoothing and velocity so a jump in position is not a "fall"
        gap = self._prev is None or t - self._prev[0] > c.max_gap
        if gap:
            self._ema = raw
        else:
            a = c.smooth_alpha
            self._ema = [r if r is None or e is None else a * r + (1 - a) * e
                         for r, e in zip(raw, self._ema)]
        hx, hy, sx, sy, ax, ay, ky = self._ema

        torso = math.hypot(sx - hx, sy - hy)
        if torso < 1.0:
            return None
        angle = math.degrees(math.atan2(abs(sx - hx), hy - sy))
        if self.ref_torso is None:
            self.ref_torso = torso
        foot_y = ay if ay is not None else ky
        hip_height = None if foot_y is None else (foot_y - hy) / self.ref_torso
        if self._upright(angle, hip_height):
            self.ref_torso = 0.9 * self.ref_torso + 0.1 * torso

        if gap:
            hip_speed = angle_rate = 0.0
        else:
            dt = max(t - self._prev[0], 1e-3)
            hip_speed = (hy - self._prev[1]) / dt / self.ref_torso
            angle_rate = abs(angle - self._prev[2]) / dt
        self._prev = (t, hy, angle)
        leg_angle = None if ay is None else math.degrees(math.atan2(abs(ax - sx), ay - sy))
        return Features(angle, hip_speed, angle_rate, hip_height, leg_angle)

    def _upright(self, angle, hip_height):
        """Torso near vertical and hips not on the floor (a torso pointing at the camera looks vertical)."""
        hips_low = hip_height is not None and hip_height <= self.cfg.hip_low_ratio
        return angle <= self.cfg.upright_angle and not hips_low

    def _enter(self, state, t):
        self.state, self.state_since = state, t

    def _step(self, f, t):
        c = self.cfg
        lo, hi = c.horizontal_angle, 180.0 - c.horizontal_angle
        flat = lo <= f.angle <= hi or (f.leg_angle is not None and lo <= f.leg_angle <= hi)
        lying = flat and (f.hip_height is None or f.hip_height <= c.hip_low_ratio)
        upright = self._upright(f.angle, f.hip_height)
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
