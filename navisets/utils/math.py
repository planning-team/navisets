import numpy as np


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Derived from
    # https://github.com/PrieureDeSion/drive-any-robot/blob/main/train/gnm_train/process_data/process_data_utils.py#L242
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw
