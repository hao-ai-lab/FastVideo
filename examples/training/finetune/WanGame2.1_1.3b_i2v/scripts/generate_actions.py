import os
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "actions"))
VIDEO_OUTPUT_DIR = BASE_OUTPUT_DIR

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

CAM_VALUE = 0.1
FRAME_COUNT = 76

# Action Mapping
KEY_TO_INDEX = {
    'W': 0, 'S': 1, 'A': 2, 'D': 3,
}

VIEW_ACTION_TO_MOUSE = {
    "stop": [0.0, 0.0],
    "up": [CAM_VALUE, 0.0],
    "down": [-CAM_VALUE, 0.0],
    "left": [0.0, -CAM_VALUE],
    "right": [0.0, CAM_VALUE],
    "up_right": [CAM_VALUE, CAM_VALUE],
    "up_left": [CAM_VALUE, -CAM_VALUE],
    "down_right": [-CAM_VALUE, CAM_VALUE],
    "down_left": [-CAM_VALUE, -CAM_VALUE],
}

def get_multihot_vector(keys_str):
    """Convert string like 'WA' to [1, 0, 1, 0, 0, 0]"""
    vector = [0.0] * 6
    if not keys_str:
        return vector
    for char in keys_str.upper():
        if char in KEY_TO_INDEX:
            vector[KEY_TO_INDEX[char]] = 1.0
    return vector

def get_mouse_vector(view_str):
    """Convert view string to [x, y]"""
    return VIEW_ACTION_TO_MOUSE.get(view_str.lower(), [0.0, 0.0])

def generate_sequence(key_seq, mouse_seq):
    """
    Generates action arrays based on sequences.
    key_seq and mouse_seq must be length FRAME_COUNT.
    Duplicates the first frame at the beginning, so output length is FRAME_COUNT + 1.
    """
    if len(key_seq) != FRAME_COUNT or len(mouse_seq) != FRAME_COUNT:
        raise ValueError("key_seq and mouse_seq must be length FRAME_COUNT")

    keyboard_arr = np.zeros((FRAME_COUNT, 6), dtype=np.float32)
    mouse_arr = np.zeros((FRAME_COUNT, 2), dtype=np.float32)

    for i in range(FRAME_COUNT):
        keyboard_arr[i] = get_multihot_vector(key_seq[i])
        mouse_arr[i] = get_mouse_vector(mouse_seq[i])

    keyboard_arr = np.vstack([keyboard_arr[0:1], keyboard_arr])
    mouse_arr = np.vstack([mouse_arr[0:1], mouse_arr])

    return keyboard_arr, mouse_arr

def save_action(filename, keyboard_arr, mouse_arr):
    if not filename.endswith(".npy"):
        filename = f"{filename}.npy"
    filepath = os.path.join(VIDEO_OUTPUT_DIR, filename)
    
    action_dict = {
        'keyboard': keyboard_arr,
        'mouse': mouse_arr
    }
    np.save(filepath, action_dict)
    return filename


def build_constant_sequence(value):
    return [value] * FRAME_COUNT


def build_random_sequence(actions, granularity, rng):
    sequence = []
    remaining = FRAME_COUNT
    while remaining > 0:
        block = granularity if remaining >= granularity else remaining
        action = rng.choice(actions)
        sequence.extend([action] * block)
        remaining -= block
    return sequence


def build_random_sequence_either_or(key_actions, mouse_actions, granularity, rng):
    """Build key_seq and mouse_seq where each block has either key OR mouse, not both."""
    key_seq = []
    mouse_seq = []
    remaining = FRAME_COUNT
    while remaining > 0:
        block = granularity if remaining >= granularity else remaining
        use_key = rng.choice([True, False])
        if use_key:
            key_action = rng.choice(key_actions)
            mouse_action = ""
        else:
            key_action = ""
            mouse_action = rng.choice(mouse_actions)
        key_seq.extend([key_action] * block)
        mouse_seq.extend([mouse_action] * block)
        remaining -= block
    return key_seq, mouse_seq


def mouse_short_name(view_str):
    mapping = {
        "up": "u",
        "down": "d",
        "left": "l",
        "right": "r",
        "up_right": "ur",
        "up_left": "ul",
        "down_right": "dr",
        "down_left": "dl",
    }
    return mapping.get(view_str, "NA")


if __name__ == "__main__":
    configs = []
    readme_content = []
    rng = np.random.default_rng(42)

    # configs = list of entries
    # a entry is a tuple of (key_seq, mouse_seq)
    # key_seq is a list of strings, length of FRAME_COUNT, each string is a key in 'W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD'
    # mouse_seq is a list of strings, length of FRAME_COUNT, each string is a mouse action in 'up', 'down', 'left', 'right', 'up_right', 'up_left', 'down_right', 'down_left'

    # Naming: 1=WASDudlr (key: W.npy, SA.npy; camera: u.npy; key+camera: W_u.npy, SA_dl.npy). 2-6=rand names below.
    # Group 1: Constant Keyboard, No Mouse. W.npy, S.npy, WA.npy, SA.npy, ...
    keys_basic = ["W", "S", "A", "D", "WA", "WD", "SA", "SD"]
    for key in keys_basic:
        configs.append(
            (key, build_constant_sequence(key), build_constant_sequence(""))
        )

    # Group 2: No Keyboard, Constant Mouse. u.npy, d.npy, ur.npy, ...
    mouse_basic = [
        "up",
        "down",
        "left",
        "right",
        "up_right",
        "up_left",
        "down_right",
        "down_left",
    ]
    for mouse in mouse_basic:
        name = mouse_short_name(mouse)
        configs.append(
            (name, build_constant_sequence(""), build_constant_sequence(mouse))
        )

    # Group 3: Still. still.npy
    configs.append(("still", build_constant_sequence(""), build_constant_sequence("")))

    # Group 4: Constant key + camera. W_u.npy, SA_dl.npy, ...
    for key in keys_basic:
        for mouse in mouse_basic:
            configs.append(
                (
                    f"{key}_{mouse_short_name(mouse)}",
                    build_constant_sequence(key),
                    build_constant_sequence(mouse),
                )
            )

    # Random groups: allow still ("") as an option (WASD+still, UDLR+still, and full sets+still)
    keys_basic_still = keys_basic + [""]
    mouse_basic_still = mouse_basic + [""]

    # Group 5: key_1_action_rand (full key set). key_1_action_rand_1..4, key_1_action_rand_1_f4..4_f4
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq = build_random_sequence(keys_basic_still, granularity, rng)
            configs.append(
                (f"key_1_action_rand_{i}{suffix}", key_seq, build_constant_sequence(""))
            )

    # Group 6: camera_1_action_rand (full camera set)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            mouse_seq = build_random_sequence(mouse_basic_still, granularity, rng)
            configs.append(
                (f"camera_1_action_rand_{i}{suffix}", build_constant_sequence(""), mouse_seq)
            )

    # Group 7: key_camera_1_action_rand (both full sets)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq = build_random_sequence(keys_basic_still, granularity, rng)
            mouse_seq = build_random_sequence(mouse_basic_still, granularity, rng)
            configs.append(
                (f"key_camera_1_action_rand_{i}{suffix}", key_seq, mouse_seq)
            )

    # WASD-only (no combined keys) and u/d/l/r-only (no combined directions), with still as option
    keys_wasd_only = ["W", "S", "A", "D"]
    mouse_udlr_only = ["up", "down", "left", "right"]
    keys_wasd_still = keys_wasd_only + [""]
    mouse_udlr_still = mouse_udlr_only + [""]

    # Group 8: key_2_action_rand (WASD+still only)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq = build_random_sequence(keys_wasd_still, granularity, rng)
            configs.append(
                (f"key_2_action_rand_{i}{suffix}", key_seq, build_constant_sequence(""))
            )

    # Group 9: camera_2_action_rand (UDLR+still only)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            mouse_seq = build_random_sequence(mouse_udlr_still, granularity, rng)
            configs.append(
                (f"camera_2_action_rand_{i}{suffix}", build_constant_sequence(""), mouse_seq)
            )

    # Group 10: key_camera_2_action_rand (WASD+still, UDLR+still)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq = build_random_sequence(keys_wasd_still, granularity, rng)
            mouse_seq = build_random_sequence(mouse_udlr_still, granularity, rng)
            configs.append(
                (f"key_camera_2_action_rand_{i}{suffix}", key_seq, mouse_seq)
            )

    # Group 11a: key_camera_excl_1_action_rand (either key OR camera per block, full key + full camera set)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq, mouse_seq = build_random_sequence_either_or(keys_basic_still, mouse_basic_still, granularity, rng)
            configs.append(
                (f"key_camera_excl_1_action_rand_{i}{suffix}", key_seq, mouse_seq)
            )

    # Group 11b: key_camera_excl_2_action_rand (either key OR camera per block, WASD/UDLR+still)
    for granularity in (4, 12):
        suffix = "_f4" if granularity == 4 else ""
        for i in range(1, 5):
            key_seq, mouse_seq = build_random_sequence_either_or(keys_wasd_still, mouse_udlr_still, granularity, rng)
            configs.append(
                (f"key_camera_excl_2_action_rand_{i}{suffix}", key_seq, mouse_seq)
            )

    # Execution
    print(f"Preparing to generate {len(configs)} action files...")

    for name, key_seq, mouse_seq in configs:
        # Generate Data
        kb_arr, ms_arr = generate_sequence(key_seq, mouse_seq)
        filename = save_action(name, kb_arr, ms_arr)
        readme_content.append(filename.replace(".npy", ""))

        print(f"Generated {filename}")

    readme_path = os.path.join(VIDEO_OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"Total Files: {len(readme_content)}\n\n")
        for idx, name in enumerate(readme_content):
            f.write(f"{idx:02d}: {name}\n")

    print(f"{len(configs)} .npy files generated in {VIDEO_OUTPUT_DIR}")