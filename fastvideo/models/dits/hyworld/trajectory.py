# HY-WorldPlay/hyvideo/generate_custom_trajectory.py

import numpy as np
import json


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def generate_camera_trajectory_local(motions):
    """
    motions: list of dict
             {"forward": 1.0}, {"yaw": np.pi/2}, {"pitch": np.pi/6}, {"right": 1.0}
             - forward: Translation (Forward or Backward)
             - yaw:   Rotate (Left or Right)
             - pitch: Rotate (Up or Down)
             - right: Translation (Right or Left)
             - third_yaw: Third Perspective Rotate (Left or Right)
    """

    poses = []
    T = np.eye(4)
    poses.append(T.copy())

    for move in motions:
        # Rotate (Left or Right)
        if "yaw" in move:
            R = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R

        # Rotate (Up or Down)
        if "pitch" in move:
            R = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R

        # Translation (Z-direction of the camera's local coordinate system)
        forward = move.get("forward", 0.0)
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Translation (Z-direction of the camera's local coordinate system)
        right = move.get("right", 0.0)
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Third Perspective Rotate (Left or Right)
        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            # Rotation around the Y-axis
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            # Translation
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            c_inv = np.linalg.inv(c_origin)
            c_relative = c_inv @ C
            T = T @ c_relative

        poses.append(T.copy())

    return poses


if __name__ == "__main__":
    # Examples: Forward 0.08 * 16 -> Right Rotate 3 degree * 16
    motions = []
    for i in range(15):
        motions.append({"forward": 0.08})

    for i in range(16):
        motions.append({"yaw": np.deg2rad(3)})

    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    poses = generate_camera_trajectory_local(motions)
    custom_c2w = {}
    for i, p in enumerate(poses):
        custom_c2w[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}
        json.dump(
            custom_c2w,
            open("./assets/pose/pose.json", "w"),
            indent=4,
            ensure_ascii=False,
        )
