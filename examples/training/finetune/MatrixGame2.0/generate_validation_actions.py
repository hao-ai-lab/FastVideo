#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np


NUM_FRAMES = 77
KEYBOARD_DIM = 23
MOUSE_DIM = 2
MOUSE_MAGNITUDE = 0.15

KEY_INDEX = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

MOUSE_VEC = {
    "left": np.array([-MOUSE_MAGNITUDE, 0.0], dtype=np.float32),
    "right": np.array([MOUSE_MAGNITUDE, 0.0], dtype=np.float32),
    "up": np.array([0.0, -MOUSE_MAGNITUDE], dtype=np.float32),
    "down": np.array([0.0, MOUSE_MAGNITUDE], dtype=np.float32),
}


def make_empty_action() -> dict[str, np.ndarray]:
    return {
        "keyboard": np.zeros((NUM_FRAMES, KEYBOARD_DIM), dtype=np.float32),
        "mouse": np.zeros((NUM_FRAMES, MOUSE_DIM), dtype=np.float32),
    }


def make_keyboard_single(key: str) -> dict[str, np.ndarray]:
    action = make_empty_action()
    action["keyboard"][:, KEY_INDEX[key]] = 1.0
    return action


def make_keyboard_transition(first: str, second: str) -> dict[str, np.ndarray]:
    action = make_empty_action()
    split = (NUM_FRAMES + 1) // 2
    action["keyboard"][:split, KEY_INDEX[first]] = 1.0
    action["keyboard"][split:, KEY_INDEX[second]] = 1.0
    return action


def make_mouse_single(direction: str) -> dict[str, np.ndarray]:
    action = make_empty_action()
    action["mouse"][:] = MOUSE_VEC[direction]
    return action


def make_mouse_transition(
    first: str, second: str
) -> dict[str, np.ndarray]:
    action = make_empty_action()
    split = (NUM_FRAMES + 1) // 2
    action["mouse"][:split] = MOUSE_VEC[first]
    action["mouse"][split:] = MOUSE_VEC[second]
    return action


def save_action(path: Path, action: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, action, allow_pickle=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "actions"

    specs: list[tuple[str, dict[str, np.ndarray]]] = [
        ("00_w_only.npy", make_keyboard_single("w")),
        ("01_s_only.npy", make_keyboard_single("s")),
        ("02_a_only.npy", make_keyboard_single("a")),
        ("03_d_only.npy", make_keyboard_single("d")),
        ("04_mouse_left_only.npy", make_mouse_single("left")),
        ("05_mouse_right_only.npy", make_mouse_single("right")),
        ("06_mouse_up_only.npy", make_mouse_single("up")),
        ("07_mouse_down_only.npy", make_mouse_single("down")),
        ("08_w_then_s.npy", make_keyboard_transition("w", "s")),
        ("09_s_then_w.npy", make_keyboard_transition("s", "w")),
        ("10_a_then_d.npy", make_keyboard_transition("a", "d")),
        ("11_d_then_a.npy", make_keyboard_transition("d", "a")),
        ("12_mouse_left_then_right.npy", make_mouse_transition("left", "right")),
        ("13_mouse_right_then_left.npy", make_mouse_transition("right", "left")),
        ("14_mouse_up_then_down.npy", make_mouse_transition("up", "down")),
        ("15_mouse_down_then_up.npy", make_mouse_transition("down", "up")),
    ]

    for filename, action in specs:
        save_action(out_dir / filename, action)
        print(f"wrote {out_dir / filename}")


if __name__ == "__main__":
    main()
