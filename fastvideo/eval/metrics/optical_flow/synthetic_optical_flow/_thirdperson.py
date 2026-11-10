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
    r_z: float = 0.0
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
