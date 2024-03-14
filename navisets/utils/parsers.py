import enum
import io
import shutil
import traceback
import numpy as np
import cv2
import h5py

from abc import ABC, abstractmethod
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from rosbags.highlevel import AnyReader
from rosbags.image import image_to_cvimage, compressed_image_to_cvimage
from navisets.utils.math import quaternion_to_yaw
from navisets.utils.strings import calculate_zeros_pad, zfill_zeros_pad
from PIL import Image


DEFAULT_IMAGE_DIR_NAME = "rgb_images"
DEFAULT_TRAJECTORY_FILE_NAME = "trajectory"
DEFAULT_IMAGE_EXTENSION = "jpg"


_TYPE_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"


class OverwritePolicy(enum.Enum):
    DELETE = "delete"
    STOP_SILENT = "stop_silent"
    RAISE = "raise"


class TopicNotFoundPolicy(enum.Enum):
    IGNORE_SILENT = "ignore_silent"
    IGNORE_LOG = "ignore_log"
    RAISE = "raise"


class AbstractDataParser(ABC):

    @abstractmethod
    def __call__(self,
                 input_path: Path,
                 output_dir: Path,
                 output_name: str | None = None) -> Path | None:
        raise NotImplementedError()


class CameraReferencedRosbagParser(AbstractDataParser):

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
                 overwrite: OverwritePolicy = OverwritePolicy.DELETE,
                 topic_not_found: TopicNotFoundPolicy = TopicNotFoundPolicy.RAISE) -> None:
        super(CameraReferencedRosbagParser, self).__init__()
        assert rate_hz > 0., f"rate_hz must be > 0., got {rate_hz}"
        self._image_topic_candidates = (image_topic_candidates,) if isinstance(
            image_topic_candidates, str) else image_topic_candidates
        self._odometry_topic_candidates = (odometry_topic_candidates,) if isinstance(
            odometry_topic_candidates, str) else odometry_topic_candidates
        self._rate = rate_hz
        self._overwrite = overwrite
        self._topic_not_found_policy = topic_not_found
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name
        self._convert_color = convert_color

    def __call__(self,
                 input_path: Path,
                 output_dir: Path,
                 output_name: str | None = None) -> Path | None:
        if not (input_path.is_file() or input_path.is_dir()):
            raise ValueError(
                f"rosbag_path {input_path} must be a file or directory")

        if output_name is None:
            output_name = CameraReferencedRosbagParser._resolve_rosbag_name(
                input_path)
        output_dir = output_dir / output_name

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

        image_data = []
        trajectory_data = []
        image_timestamps = []
        trajectory_timestamps = []
        dt = 1 / self._rate

        with AnyReader([input_path]) as reader:
            image_topic, odometry_topic = self._filter_topics(reader)
            if image_topic is None:
                self._handle_topic_not_found(self._image_topic_candidates, input_path)
                return None
            if odometry_topic is None:
                self._handle_topic_not_found(self._odometry_topic_candidates, input_path)          
                return None
            connections = [e for e in reader.connections if e.topic in (
                image_topic,
                odometry_topic
            )]

            output_images_dir = output_dir / self._images_dir_name
            output_images_dir.mkdir(parents=True, exist_ok=False)

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

        n_zeros = calculate_zeros_pad(len(image_data))
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

    def _handle_topic_not_found(self, candidates: tuple[str], rosbag_path: Path) -> None:
        match self._topic_not_found_policy:
            case TopicNotFoundPolicy.IGNORE_SILENT:
                return
            case TopicNotFoundPolicy.IGNORE_LOG:
                print(f"Unable to find any of candidate topic among {candidates} in rosbag {rosbag_path}")
            case TopicNotFoundPolicy.RAISE:
                raise RuntimeError(f"Unable to find any of candidate topic among {candidates} in rosbag {rosbag_path}")
            case _:
                raise ValueError(f"Unknown topic not found policy {self._topic_not_found_policy}")

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


class HDF5FileParser(AbstractDataParser):
    
    _HDF5_EXTENSION = "hdf5"
    
    def __init__(self,
                 images_dir_name: str = DEFAULT_IMAGE_DIR_NAME,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 trajectory_file_name: str = DEFAULT_TRAJECTORY_FILE_NAME,
                 overwrite: OverwritePolicy = OverwritePolicy.DELETE) -> None:
        super(HDF5FileParser, self).__init__()
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name
        self._overwrite = overwrite

    def __call__(self, input_path: Path, output_dir: Path, output_name: str | None = None) -> Path | None:
        if not input_path.is_file():
            raise ValueError(f"Input path {input_path} muse be a file")
        name_parts = input_path.name.split(".")
        if name_parts[-1] != HDF5FileParser._HDF5_EXTENSION:
            raise ValueError(f"Input file must have .{HDF5FileParser._HDF5_EXTENSION} extension, got {input_path}")
        if output_name is None:
            output_name = "".join(name_parts[:-1])
        output_dir = output_dir / output_name

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
        
        hdf5_file = h5py.File(str(input_path), "r")
        
        position_data = hdf5_file["jackal"]["position"][:, :2]
        yaw_data = hdf5_file["jackal"]["yaw"][()]
        images_data = hdf5_file["images"]["rgb_left"]
        n_frames = images_data.shape[0]
        n_zeros = calculate_zeros_pad(n_frames)

        if position_data.shape[0] != yaw_data.shape[0]:
            raise RuntimeError(f"Position and yaw lengths do not match for {input_path}")
        if position_data.shape[0] != n_frames:
            raise RuntimeError(f"Position data and image data do not match for {input_path}")

        trajectory = []
        for i in range(n_frames):
            image = Image.open(io.BytesIO(images_data[i]))
            image_file = output_images_dir / f"{zfill_zeros_pad(i, n_zeros)}.{self._image_extension}"
            image.save(str(image_file))
            trajectory.append([position_data[i, 0], position_data[i, 1], yaw_data[i]])
        
        trajectory = np.array(trajectory)
        np.save(str(output_dir / self._trajectory_file_name), trajectory)
        
        return output_dir


class MulitprocessParseWrapper:

    N_WORKERS_NO_MULTIPROCESS = 0

    def __init__(self,
                 parser: AbstractDataParser,
                 n_workers: int) -> None:
        assert isinstance(
            n_workers, int) and n_workers >= 0, f"n_workers must be int >= 0, got {n_workers}"
        self._parser = parser
        self._n_workers = n_workers

    def __call__(self, input_paths: list[Path], output_parent_dir: Path) -> list[Path | None]:
        process_fn = partial(self._do_parse, output_dir=output_parent_dir)

        if len(input_paths) == 1:
            return [process_fn(input_paths[0])]

        if self._n_workers == MulitprocessParseWrapper.N_WORKERS_NO_MULTIPROCESS:
            return [process_fn(e) for e in input_paths]

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            result = executor.map(process_fn, input_paths)
            return [e for e in result]
        
    def _do_parse(self, input_path: Path, output_dir: Path) -> Path | None:
        result = None
        try:
            result = self._parser(input_path, output_dir)
        except Exception as e:
            print(f"Exception handled when parsing {input_path}")
            print(traceback.format_exc())
        return result
