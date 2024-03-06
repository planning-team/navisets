import enum
import os
import shutil
import numpy as np
import cv2

from abc import ABC, abstractmethod
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from rosbags.highlevel import AnyReader
from rosbags.image import image_to_cvimage, compressed_image_to_cvimage
from navisets.utils.math import quaternion_to_yaw
from navisets.utils.strings import calculate_zeros_pad, zfill_zeros_pad


_TYPE_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"


class OverwritePolicy(enum.Enum):
    DELETE = "delete"
    STOP_SILENT = "stop_silent"
    RAISE = "raise"


class AbstractRosbagParser(ABC):

    @abstractmethod
    def __call__(self,
                 rosbag_path: Path,
                 output_dir: Path,
                 rosbag_name: str | None = None) -> Path | None:
        raise NotImplementedError()


class CameraReferencedRosbagParser(AbstractRosbagParser):

    DEFAULT_IMAGE_DIR_NAME = "rgb_images"
    DEFAULT_TRAJECTORY_FILE_NAME = "trajectory"
    DEFAULT_IMAGE_EXTENSION = "jpg"

    _EXTENSION_BAG = "bag"
    _TEMP_PREFIX = "temp_"

    def __init__(self,
                 rate_hz: float,
                 image_topic_candidates: str | tuple[str],
                 odometry_topic_candidates: str | tuple[str],
                 images_dir_name: str = DEFAULT_IMAGE_DIR_NAME,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 trajectory_file_name: str = DEFAULT_TRAJECTORY_FILE_NAME,
                 convert_color: bool = True,
                 overwrite: OverwritePolicy = OverwritePolicy.DELETE) -> None:
        super(CameraReferencedRosbagParser, self).__init__()
        assert rate_hz > 0., f"rate_hz must be > 0., got {rate_hz}"
        self._image_topic_candidates = (image_topic_candidates,) if isinstance(
            image_topic_candidates, str) else image_topic_candidates
        self._odometry_topic_candidates = (odometry_topic_candidates,) if isinstance(
            odometry_topic_candidates, str) else odometry_topic_candidates
        self._rate = rate_hz
        self._overwrite = overwrite
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name
        self._convert_color = convert_color

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
                    raise RuntimeError(f"Output directory {output_dir} already exists")
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
            image_topic, odometry_topic = self._filter_topics(reader)
            if image_topic is None:
                raise RuntimeError(f"Unable to find any of candidate image topic {self._image_topic_candidates} in rosbag {rosbag_path}")
            if odometry_topic is None:
                raise RuntimeError(f"Unable to find any of candidate odometry topic {self._odometry_topic_candidates} in rosbag {rosbag_path}")             
            connections = [e for e in reader.connections if e.topic in (
                image_topic,
                odometry_topic
            )]

            last_image_timestamp = None
            image_count = 0

            for connection, timestamp, rawdata in reader.messages(connections=connections):

                if connection.topic == image_topic:
                    if last_image_timestamp is None or (timestamp - last_image_timestamp) / 1e9 >= dt:
                        image_timestamps.append(timestamp)
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        if connection.msgtype == _TYPE_COMPRESSED_IMAGE:
                            img = compressed_image_to_cvimage(msg)
                        else:
                            img = image_to_cvimage(msg)
                        if self._convert_color:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_path = output_images_dir / \
                            f"{CameraReferencedRosbagParser._TEMP_PREFIX}{
                                image_count}.{self._image_extension}"
                        cv2.imwrite(str(img_path), img)
                        image_data.append(img_path)
                        image_count += 1
                        last_image_timestamp = timestamp

                elif connection.topic == odometry_topic:
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

        odometry_indices = CameraReferencedRosbagParser._sync_timestamps(
            image_timestamps, trajectory_timestamps)
        trajectory_data = trajectory_data[odometry_indices]

        n_zeros = calculate_zeros_pad(image_data)
        for i, img_path in enumerate(image_data):
            new_path = img_path.parent / \
                f"{zfill_zeros_pad(i, n_zeros)}.{self._image_extension}"
            img_path.rename(new_path)
        np.save(str(output_dir / self._trajectory_file_name),
                np.array(trajectory_data))

        return output_dir

    def _filter_topics(self, reader: AnyReader) -> tuple[str | None, str | None]:
        image_topic = None
        odometry_topic = None
        reader_topics = reader.topics
        for topic in self._image_topic_candidates:
            if topic in reader_topics:
                image_topic = topic
                break
        for topic in self._odometry_topic_candidates:
            if topic in reader_topics:
                odometry_topic = topic
                break
        return image_topic, odometry_topic

    @staticmethod
    def _resolve_rosbag_name(rosbag_path: Path) -> str | None:
        if rosbag_path.is_file():
            name_parts = rosbag_path.name.split(".")
            if name_parts[-1] == CameraReferencedRosbagParser._EXTENSION_BAG:
                return "".join(name_parts[:-1])
            else:
                return "".join(name_parts)
        elif rosbag_path.is_dir():
            return rosbag_path.name
        return None

    @staticmethod
    def _sync_timestamps(anchor_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
        diffs = np.abs(anchor_ts[:, np.newaxis] - source_ts[np.newaxis, :])
        indices = np.argmin(diffs, axis=1)
        return indices


class MulitprocessRosbagParseWrapper:

    N_WORKERS_NO_MULTIPROCESS = 0

    def __init__(self,
                 parser: AbstractRosbagParser,
                 n_workers: int) -> None:
        assert isinstance(
            n_workers, int) and n_workers >= 0, f"n_workers must be int >= 0, got {n_workers}"
        self._parser = parser
        self._n_workers = n_workers

    def __call__(self, rosbag_dirs: list[Path], output_parent_dir: Path) -> list[Path | None]:
        process_fn = partial(self._parser, output_dir=output_parent_dir)

        if len(rosbag_dirs) == 1:
            return [process_fn(rosbag_dirs[0])]

        if self._n_workers == MulitprocessRosbagParseWrapper.N_WORKERS_NO_MULTIPROCESS:
            return [process_fn(e) for e in rosbag_dirs]

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            return [e for e in executor.map(process_fn, rosbag_dirs)]
