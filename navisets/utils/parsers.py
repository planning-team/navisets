import enum
import os
import shutil
import numpy as np
import cv2

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import image_to_cvimage
from navisets.utils.math import quaternion_to_yaw
from navisets.utils.strings import calculate_zeros_pad, zfill_zeros_pad


class OverwritePolicy(enum.Enum):
    DELETE = "delete"
    STOP_SILENT = "stop_silent"
    RAISE = "raise"


class CameraReferencedRosbagParser:

    DEFAULT_IMAGE_DIR_NAME = "rgb_images"
    DEFAULT_TRAJECTORY_FILE_NAME = "trajectory"
    DEFAULT_IMAGE_EXTENSION = "jpg"

    _EXTENSION_BAG = "bag"
    _TEMP_PREFIX = "temp_"

    def __init__(self,
                 rate_hz: float,
                 image_topic: str,
                 odometry_topic: str,
                 images_dir_name: str = DEFAULT_IMAGE_DIR_NAME,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 trajectory_file_name: str = DEFAULT_TRAJECTORY_FILE_NAME,
                 overwrite: OverwritePolicy = OverwritePolicy.DELETE) -> None:
        assert rate_hz > 0., f"rate_hz must be > 0., got {rate_hz}"
        self._image_topic = image_topic
        self._odometry_topic = odometry_topic
        self._rate = rate_hz
        self._overwrite = overwrite
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name

    def __call__(self,
                 rosbag_path: Path,
                 output_dir: Path,
                 rosbag_name: str | None = None) -> Path | None:
        if not (rosbag_path.is_file() or rosbag_path.is_dir()):
            raise ValueError(
                f"rosbag_path {rosbag_path} must be a file or directory")

        if rosbag_name is None:
            rosbag_name = CameraReferencedRosbagParser._resolve_rosbag_name(
                rosbag_path)
        output_dir = output_dir / rosbag_name

        if output_dir.is_dir():
            match self._overwrite:
                case OverwritePolicy.DELETE:
                    shutil.rmtree(str(output_dir))
                case OverwritePolicy.STOP_SILENT:
                    return None
                case OverwritePolicy.RAISE:
                    raise RuntimeError(f"Output directory {
                                       output_dir} already exists")
                case _:
                    raise ValueError(f"Unknown overwrite policy")
        output_images_dir = output_dir / self._images_dir_name
        output_images_dir.mkdir(parents=True, exist_ok=False)

        image_data = []
        trajectory_data = []
        image_timestamps = []
        trajectory_timestamps = []
        dt = 1 / self._rate

        with AnyReader([rosbag_path]) as reader:
            connections = [e for e in reader.connections if e.topic in (
                self._image_topic,
                self._odometry_topic
            )]

            last_image_timestamp = None
            image_count = 0

            for connection, timestamp, rawdata in reader.messages(connections=connections):

                if connection.topic == self._image_topic:
                    if last_image_timestamp is None or (timestamp - last_image_timestamp) / 1e9 >= dt:
                        image_timestamps.append(timestamp)
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        img = image_to_cvimage(msg)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_path = output_images_dir / \
                            f"{CameraReferencedRosbagParser._TEMP_PREFIX}{
                                image_count}.{self._image_extension}"
                        cv2.imwrite(str(img_path), img)
                        image_data.append(img_path)
                        image_count += 1
                        last_image_timestamp = timestamp

                elif connection.topic == self._odometry_topic:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    trajectory_data.append([msg.pose.pose.position.x, 
                                            msg.pose.pose.position.y,
                                            quaternion_to_yaw(msg.pose.pose.orientation.x,
                                                              msg.pose.pose.orientation.y,
                                                              msg.pose.pose.orientation.z,
                                                              msg.pose.pose.orientation.w)])
                    trajectory_timestamps.append(timestamp)

        image_timestamps = np.array(image_timestamps)
        trajectory_data = np.array(trajectory_data)
        trajectory_timestamps = np.array(trajectory_timestamps)

        odometry_indices = CameraReferencedRosbagParser._sync_timestamps(image_timestamps, trajectory_timestamps)
        trajectory_data = trajectory_data[odometry_indices]

        n_zeros = calculate_zeros_pad(image_data)
        for i, img_path in image_data:
            new_path = img_path.parent / f"{zfill_zeros_pad(i, n_zeros)}.{self._image_extension}"
            img_path.rename(new_path)
        np.save(str(output_dir / self._trajectory_file_name), np.array(trajectory_data))

        return output_dir

    @staticmethod
    def _resolve_rosbag_name(rosbag_path: Path) -> str | None:
        if rosbag_path.is_file():
            extension = rosbag_path.name.split(".")[-1]
            if extension == CameraReferencedRosbagParser._EXTENSION_BAG:
                return "".join(extension[:-1])
            else:
                return "".join(extension)
        elif rosbag_path.is_dir():
            return rosbag_path.name
        return None

    @staticmethod
    def _sync_timestamps(anchor_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
        diffs = np.abs(anchor_ts[:, np.newaxis] - source_ts[np.newaxis, :])
        indices = np.argmin(diffs, axis=1)
        return indices
