"""Third-person synthetic optical-flow generator (Option B, no depth).

Camera frame convention (OpenCV): x=right, y=down, z=forward.

Per-frame inputs from the action stream:
    keyboard : (6,) — [W, S, A, D, turn_left, turn_right]
    mouse    : (2,) — [pitch, yaw]   (pitch may be sign-flipped per sample,
                                      see ``mouse_pitch_sign``)

Mapping to camera kinematics:
    omega_x = alpha_pitch * mouse_pitch                       (camera pitch)
    omega_y = alpha_yaw   * mouse_yaw
            + alpha_turn  * (turn_right - turn_left)          (camera yaw)
    omega_z = 0                                               (no roll)

    T_avatar_x = beta_strafe * (D - A)
    T_avatar_y = 0
    T_avatar_z = beta_fwd    * (W - S)

Off-pivot correction. The orbit camera rotates about the avatar pivot at
``r = (0, r_y, r_z)`` in camera-local coords, not about the optical
center. A rotation by omega about that pivot is kinematically equivalent
to a rotation about the optical center plus a translation
``T_orbit = -(omega x r)``. So:

    T_total = T_avatar + T_orbit

Flow (no depth, Z = 1):
    u_R = (xy/f)*ωx - (f + x²/f)*ωy + y*ωz
    v_R = (f + y²/f)*ωx - (xy/f)*ωy - x*ωz
    u_T = -f*Tx + x*Tz
    v_T = -f*Ty + y*Tz

The Z=1 collapse means strafe (Tx-only) produces a uniform horizontal
field. Forward motion (Tz) still has the right radial direction
structure since u_T scales with x, just no depth-modulated magnitude.
Angle-family metrics survive this; per-pixel magnitude metrics will be
biased on parallax-rich backgrounds. That's the explicit cost of
declining to use generated-video depth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ThirdPersonCalibration:
    """Fitted parameters for a third-person rig.

    ``r_y`` defaults to 0 (avatar at camera height). ``focal_length`` is
    in pixels. ``init_pitch`` is the camera's rest pitch (radians, negative
    means tilted down — typical over-the-shoulder framing). User mouse-pitch
    input is integrated on top of this baseline.
    """
    alpha_yaw: float
    alpha_pitch: float
    alpha_turn: float
    beta_fwd: float
    beta_strafe: float
    focal_length: float
    r_z: float
    r_y: float = 0.0
    init_pitch: float = 0.0
    notes: str = ""
    fit_metadata: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> ThirdPersonCalibration:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_calibration(path: str | Path) -> ThirdPersonCalibration:
    return ThirdPersonCalibration.from_dict(json.loads(Path(path).read_text()))


class ThirdPersonFlowGenerator:
    """Vectorized 3P synthetic-flow generator. No depth.

    Parameters
    ----------
    calibration : ThirdPersonCalibration
    frame_shape : (H, W)
    mouse_pitch_sign : +1 or -1 — sample-level flag from metadata
        (``mouse_pitch_flipped: true`` in mhuo's data ⇒ -1).
    """

    def __init__(
        self,
        calibration: ThirdPersonCalibration,
        frame_shape: tuple[int, int],
        mouse_pitch_sign: int = +1,
    ) -> None:
        self.cal = calibration
        self.H, self.W = frame_shape
        self.mouse_pitch_sign = int(mouse_pitch_sign)

        f = self.cal.focal_length
        cx, cy = self.W / 2.0, self.H / 2.0
        xs = np.arange(self.W, dtype=np.float64) - cx
        ys = np.arange(self.H, dtype=np.float64) - cy
        self.x_grid, self.y_grid = np.meshgrid(xs, ys)  # H,W

        # Pre-compute LH rotation kernels (depend only on pixel coords + f).
        self.xy_over_f = self.x_grid * self.y_grid / f
        self.f_plus_x2_over_f = f + self.x_grid**2 / f
        self.f_plus_y2_over_f = f + self.y_grid**2 / f

    @staticmethod
    def _action_to_kinematics(
        keyboard: np.ndarray,
        mouse: np.ndarray,
        cal: ThirdPersonCalibration,
        mouse_pitch_sign: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (omega (3,), T_total (3,)) for one frame's action."""
        kb = np.asarray(keyboard, dtype=np.float64).reshape(-1)
        mo = np.asarray(mouse, dtype=np.float64).reshape(-1)

        pitch = mo[0] * mouse_pitch_sign
        yaw = mo[1]

        omega_x = cal.alpha_pitch * pitch
        omega_y = cal.alpha_yaw * yaw
        if kb.shape[0] >= 6:
            omega_y += cal.alpha_turn * (kb[5] - kb[4])
        omega = np.array([omega_x, omega_y, 0.0])

        T_avatar = np.array([
            cal.beta_strafe * (kb[3] - kb[2]),
            0.0,
            cal.beta_fwd * (kb[0] - kb[1]),
        ])

        # T_orbit = -(omega x r) with r = (0, r_y, r_z).
        # cross([wx,wy,0], [0,ry,rz]) = (wy*rz, -wx*rz, wx*ry)
        T_orbit = -np.array([
            omega[1] * cal.r_z,
            -omega[0] * cal.r_z,
            omega[0] * cal.r_y,
        ])
        return omega, T_avatar + T_orbit

    def _flow_from_kinematics(self, omega: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compose rotation + translation flow at every pixel. Z=1."""
        wx, wy, wz = omega
        Tx, Ty, Tz = T
        f = self.cal.focal_length

        u_R = self.xy_over_f * wx - self.f_plus_x2_over_f * wy + self.y_grid * wz
        v_R = self.f_plus_y2_over_f * wx - self.xy_over_f * wy - self.x_grid * wz
        u_T = -f * Tx + self.x_grid * Tz
        v_T = -f * Ty + self.y_grid * Tz
        return np.stack([u_R + u_T, v_R + v_T], axis=-1).astype(np.float32)

    def generate_flow(
        self,
        keyboard: np.ndarray,
        mouse: np.ndarray,
    ) -> np.ndarray:
        """Synthesize HxWx2 flow for one frame's action."""
        omega, T = self._action_to_kinematics(
            keyboard,
            mouse,
            self.cal,
            self.mouse_pitch_sign,
        )
        return self._flow_from_kinematics(omega, T)

    def generate_flow_sequence(
        self,
        actions: dict,
        n_pairs: int | None = None,
    ) -> list[np.ndarray]:
        """Generate flow for each consecutive frame pair.

        Returns ``n`` flows where ``n = n_pairs`` if supplied, else
        ``len(keyboard) - 1``.
        """
        kb = actions["keyboard"]
        mo = actions["mouse"]
        T = len(kb)
        n = (T - 1) if n_pairs is None else min(n_pairs, T - 1)
        return [self.generate_flow(kb[i], mo[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Linear-features form (used by the calibration fitter).
# ---------------------------------------------------------------------------

# Number of free params we fit. Order is fixed and shared with the fitter.
PARAM_NAMES = (
    "alpha_yaw",
    "alpha_pitch",
    "alpha_turn",
    "beta_fwd",
    "beta_strafe",
    "focal_length",
    "r_z",
    "r_y",
)


def predict_flow_at_pixels(
    keyboard: np.ndarray,  # (6,)
    mouse: np.ndarray,  # (2,)
    xs_centered: np.ndarray,  # (N,) pixel x relative to principal point
    ys_centered: np.ndarray,  # (N,)
    cal: ThirdPersonCalibration,
    mouse_pitch_sign: int,
) -> np.ndarray:
    """Vectorized flow prediction at an arbitrary pixel set. Returns (N, 2).

    Stateless: assumes camera is level (theta_pitch=0). For 3P games where
    the camera tilts independently of the avatar's facing, use
    :func:`predict_flow_at_pixels_stateful` and pass the integrated pitch.
    """
    omega, T = ThirdPersonFlowGenerator._action_to_kinematics(
        keyboard,
        mouse,
        cal,
        mouse_pitch_sign,
    )
    wx, wy, wz = omega
    Tx, Ty, Tz = T
    f = cal.focal_length

    xy_over_f = xs_centered * ys_centered / f
    f_plus_x2_over_f = f + xs_centered**2 / f
    f_plus_y2_over_f = f + ys_centered**2 / f

    u_R = xy_over_f * wx - f_plus_x2_over_f * wy + ys_centered * wz
    v_R = f_plus_y2_over_f * wx - xy_over_f * wy - xs_centered * wz
    u_T = -f * Tx + xs_centered * Tz
    v_T = -f * Ty + ys_centered * Tz
    return np.stack([u_R + u_T, v_R + v_T], axis=-1)


def predict_flow_at_pixels_stateful(
    keyboard: np.ndarray,
    mouse: np.ndarray,
    xs_centered: np.ndarray,
    ys_centered: np.ndarray,
    cal: ThirdPersonCalibration,
    theta_pitch: float,
) -> np.ndarray:
    """Stateful flow prediction that accounts for accumulated camera pitch.

    In a 3P game the avatar moves in the world's horizontal plane in the
    direction the camera is yawed. When the camera is also pitched
    (looking down at the avatar / up at the sky), this world-horizontal
    motion has a non-zero y component in the camera frame. Concretely:

        T_cam = β_fwd · (W − S) · (0, sin θ_pitch, cos θ_pitch)
              + β_strafe · (D − A) · (1, 0, 0)              # strafe is pitch-invariant

    Yaw doesn't appear because the camera frame is yaw-aligned by construction
    (avatar and camera yaw together). Mouse rotation contributions to ω are
    unchanged.

    Parameters
    ----------
    theta_pitch : float (radians)
        Accumulated camera pitch state at this frame, integrated from
        prior mouse-pitch input. Positive = camera looking up.
    """
    f = cal.focal_length
    cos_p = float(np.cos(theta_pitch))
    sin_p = float(np.sin(theta_pitch))

    # Rotational velocities (unchanged from stateless)
    pitch_in = float(mouse[0])
    yaw_in = float(mouse[1])
    omega_x = cal.alpha_pitch * pitch_in
    omega_y = cal.alpha_yaw * yaw_in
    if keyboard.shape[0] >= 6:
        omega_y += cal.alpha_turn * (keyboard[5] - keyboard[4])

    # Avatar-frame translations (in world-horizontal plane, avatar-yaw-aligned)
    avatar_strafe = cal.beta_strafe * (keyboard[3] - keyboard[2])
    avatar_fwd = cal.beta_fwd * (keyboard[0] - keyboard[1])

    # Map to camera frame using current pitch
    Tx = avatar_strafe
    Ty = avatar_fwd * sin_p
    Tz = avatar_fwd * cos_p

    xy_over_f = xs_centered * ys_centered / f
    f_plus_x2_over_f = f + xs_centered**2 / f
    f_plus_y2_over_f = f + ys_centered**2 / f

    u_R = xy_over_f * omega_x - f_plus_x2_over_f * omega_y
    v_R = f_plus_y2_over_f * omega_x - xy_over_f * omega_y
    u_T = -f * Tx + xs_centered * Tz
    v_T = -f * Ty + ys_centered * Tz
    return np.stack([u_R + u_T, v_R + v_T], axis=-1)


def integrate_pitch_state(
    mouse: np.ndarray,  # (T, 2) raw or cached actions
    cal: ThirdPersonCalibration,
    *,
    init_pitch: float = 0.0,
    frames_per_step: int = 1,
) -> np.ndarray:
    """Integrate per-frame mouse-pitch input into accumulated camera pitch.

    Returns a (T,) array where ``out[t]`` is the camera's accumulated pitch
    angle (radians) AT THE START of frame ``t`` — i.e. the pose under which
    frame ``t``'s action is interpreted.

    For raw-frame action sequences pass ``frames_per_step=1``. For cached
    actions where each sample represents N raw frames of integration
    (cache stride = N), pass ``frames_per_step=N``.

    NOTE: this only models the explicit user-input pitch. Cinematic
    auto-pitch (camera tilting to track the avatar over uneven terrain)
    isn't in the action stream and isn't captured here. For that you need
    visual odometry (Option B / WorldCam-style ViPE pipeline).
    """
    per_step = cal.alpha_pitch * np.asarray(mouse[:, 0], dtype=np.float64) * frames_per_step
    cum = np.cumsum(per_step)
    # out[t] = pose BEFORE frame t's input is applied → shift by one
    return init_pitch + np.concatenate([[0.0], cum[:-1]]).astype(np.float64)
