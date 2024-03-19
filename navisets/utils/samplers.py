import enum
import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from navisets.constants import DEFAULT_IMAGE_DIR_NAME, DEFAULT_IMAGE_EXTENSION, DEFAULT_TRAJECTORY_FILE_NAME
from navisets.utils.random import get_or_sample_int
from navisets.utils.math import to_relative_frame


class IncompleteTrajectory(enum.Enum):
    IGNORE = "ignore"
    WARN = "warn"
    RAISE = "raise"


class AbstractEntitySampler(ABC):

    @abstractmethod    
    def __call__(self, input_trajectory: Path) -> list[dict[str, Any]]:
        raise NotImplementedError()


@dataclass
class ContextActionTrajectorySample:
    context: tuple[str]
    observation: str
    goal: str
    action: tuple[tuple[float, float]]

    
class ContextActionTrajectorySampler(AbstractEntitySampler):
    
    def __init__(self,
                 context_length: int,
                 action_length: int,
                 goal_offset: int | tuple[int, int],
                 n_discard_first_frames: int = 0,
                 n_discard_last_frames: int = 0,
                 sample_interval: int = 1,
                 incomplete_policy: IncompleteTrajectory = IncompleteTrajectory.WARN,
                 images_dir_name: str = DEFAULT_IMAGE_DIR_NAME,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 trajectory_file_name: str = DEFAULT_TRAJECTORY_FILE_NAME):
        super().__init__()
        self._context_length = context_length
        self._action_length = action_length
        self._goal_offset = goal_offset
        self._n_discard_first_frames = n_discard_first_frames
        self._n_discard_last_frames = n_discard_last_frames
        self._sample_interval = sample_interval
        self._incomplete_policy = incomplete_policy
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name
    
    def __call__(self, input_trajectory: Path) -> list[dict[str, Any]]:
        trajectory_file = input_trajectory / self._trajectory_file_name
        images_dir = input_trajectory / self._images_dir_name
        if not trajectory_file.is_file():
            return self._process_error(f"Trajectory file {trajectory_file} does not exist")
        if not images_dir.is_dir():
            return self._process_error(f"Images directory {images_dir} does not exist")
        
        trajectory = np.load(str(trajectory_file))
        if len(trajectory.shape) != 2 and trajectory.shape[1] != 3: 
            return self._process_error(f"Trajectory has unsupported shape {trajectory.shape} in {input_trajectory}")
        images = sorted(images_dir.glob(f"*.{self._image_extension}"))
        if len(images) != trajectory.shape[0]:
            return self._process_error(f"Trajectory length {trajectory.shape[0]} does not match number of images {len(images)} in {input_trajectory}")
        
        start_offset = self._n_discard_first_frames + self._context_length
        end_offset = None if self._n_discard_last_frames == 0 else self._n_discard_last_frames
        trajectory = trajectory[start_offset:end_offset]
        images = trajectory[start_offset:end_offset]

        samples = []

    def _sample(self, full_trajectory: np.ndarray, full_images: list[Path], start_idx: int) -> ContextActionTrajectorySample:
        goal_idx = start_idx + get_or_sample_int(self._goal_offset)
        if goal_idx >= full_trajectory.shape[0]:
            goal_idx = full_trajectory.shape[0] - 1
        
        context = (str(e) for e in full_images[start_idx - self._context_length:start_idx])
        observation = str(full_images[start_idx])
        goal = str(full_images[goal_idx])

        sampled_action_length = min(goal_idx - start_idx, self._action_length)
        if sampled_action_length != 0:
            action = full_trajectory[start_idx:(start_idx + sampled_action_length + 1)]
            action = to_relative_frame(action)
            action = action[:, :2]
            action = ((e[0], e[1]) for e in action)
        else:
            action = ((0., 0.) for _ in range(self._action_length))

        sample = ContextActionTrajectorySample(context=context,
                                               observation=observation,
                                               goal=goal,
                                               action=action)
        return sample


    def _process_error(self, message: str) -> list[dict[str, Any]]:
        match self._incomplete_policy:
            case IncompleteTrajectory.IGNORE:
                return []
            case IncompleteTrajectory.WARN:
                print(message)
                return []
            case IncompleteTrajectory.RAISE:
                raise RuntimeError(message)
            case _:
                raise ValueError(f"Unknown incomplete trajectory policy {self._incomplete_policy}")
