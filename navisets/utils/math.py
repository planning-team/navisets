import numpy as np


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Derived from
    # https://github.com/PrieureDeSion/drive-any-robot/blob/main/train/gnm_train/process_data/process_data_utils.py#L242
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def wrap_angle(angle: np.ndarray | float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def to_relative_frame(poses: np.ndarray, reference_pose: np.ndarray | None = None) -> np.ndarray:
    assert (len(poses.shape) == 2 and poses.shape[1] == 3) or poses.shape == (3,), f"poses must have shape (N, 3) or (3,), got {poses.shape}"
    if len(poses.shape) == 1:
        poses = poses[np.newaxis, :]
        single_pose = True
    else:
        single_pose = False
    if reference_pose is not None:
        assert reference_pose.shape == (3,), f"Reference pose must have shape (3,), got {reference_pose.shape}"
    else:
        reference_pose = poses[0]
    
    new_poses = np.zeros_like(poses)
    reference_yaw = wrap_angle(reference_pose[2])
    new_poses[:, 2] = wrap_angle(poses[:, 2] - reference_yaw)
    new_poses[:, :2] = poses[:, :2] - reference_pose[:2]
    rotation_matrix = np.array([[np.cos(-reference_yaw), -np.sin(-reference_yaw)],
                                [np.sin(-reference_yaw), np.cos(-reference_yaw)]])
    # for i in range(new_poses.shape[0]):
    #     new_poses[i, :2] = rotation_matrix @ new_poses[i, :2]
    new_poses[:, :2] = np.einsum("ji,ni ->nj", rotation_matrix, new_poses[:, :2])

    if single_pose:
        new_poses = new_poses[0]

    return new_poses
