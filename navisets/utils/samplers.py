import enum
import traceback
import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from navisets.constants import DEFAULT_IMAGE_DIR_NAME, DEFAULT_IMAGE_EXTENSION, DEFAULT_TRAJECTORY_FILE_NAME
from navisets.utils.random import get_or_sample_int
from navisets.utils.math import to_relative_frame


class IncompleteTrajectory(enum.Enum):
    IGNORE = "ignore"
    WARN = "warn"
    RAISE = "raise"


class AbstractEntitySampler(ABC):

    @abstractmethod    
    def __call__(self, input_trajectory: Path, all_trajectories: list[Path] | None = None) -> list[Any]:
        raise NotImplementedError()


# @dataclass
# class ContextActionTrajectorySample:
#     context: tuple[str]
#     observation: str
#     goal: str
#     action: tuple[tuple[float, float]]
    
# ContextActionTrajectorySample = 

    
class ContextActionTrajectorySampler(AbstractEntitySampler):
    
    def __init__(self,
                 context_length: int,
                 action_length: int,
                 goal_offset: tuple[int, int],
                 negative_mining: bool,
                 n_discard_first_frames: int = 0,
                 n_discard_last_frames: int = 0,
                 sample_interval: int = 1,
                 negative_probability: float | None = None,
                 incomplete_policy: IncompleteTrajectory = IncompleteTrajectory.WARN,
                 images_dir_name: str = DEFAULT_IMAGE_DIR_NAME,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 trajectory_file_name: str = DEFAULT_TRAJECTORY_FILE_NAME):
        super().__init__()
        self._context_length = context_length
        self._action_length = action_length
        self._goal_offset = goal_offset
        self._negative_mining = negative_mining
        self._n_discard_first_frames = n_discard_first_frames
        self._n_discard_last_frames = n_discard_last_frames
        self._sample_interval = sample_interval
        self._incomplete_policy = incomplete_policy
        self._images_dir_name = images_dir_name
        self._image_extension = image_extension
        self._trajectory_file_name = trajectory_file_name
        if negative_mining:
            if negative_probability is not None:
                self._negative_probability = negative_probability
            else:
                self._negative_probability = 1. / (goal_offset[1] - goal_offset[0] + 2)
        else:
            self._negative_probability = None
    
    def __call__(self, input_trajectory: Path, all_trajectories: list[Path] | None = None) -> list[tuple[Any]]:
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
        
        start_offset = self._n_discard_first_frames
        end_offset = None if self._n_discard_last_frames == 0 else -self._n_discard_last_frames
        trajectory = trajectory[start_offset:end_offset]
        images = images[start_offset:end_offset]

        samples = []
        for i in range(self._context_length, trajectory.shape[0], self._sample_interval):
            sample = self._sample(trajectory, images, i, all_trajectories)
            samples.append(sample)

        return samples

    def _sample(self, full_trajectory: np.ndarray, full_images: list[Path], start_idx: int, 
                all_trajectories: list[Path] | None = None) -> tuple[Any]:
        negative_mining = self._negative_mining and (all_trajectories is not None)
        if negative_mining:
            negative_goal = np.random.uniform(0., 1.) < self._negative_probability
        else:
            negative_goal = False

        if negative_goal:
            goal = self._sample_negative_goal_image(all_trajectories)
            goal_distance = self._goal_offset[1] + 2
        else:
            goal_offsets = sorted(range(self._goal_offset[0], self._goal_offset[1] + 1))
            goal_offset = np.random.choice(goal_offsets)
            goal_offset = min(goal_offset, full_trajectory.shape[0] - start_idx - 1)
            goal_idx = start_idx + goal_offset
            goal = str(full_images[goal_idx])
            goal_distance = goal_offset
        
        context = tuple([str(e) for e in full_images[start_idx - self._context_length:start_idx]])
        observation = str(full_images[start_idx])

        sampled_action_length = min(goal_idx - start_idx, self._action_length) if not negative_goal else 0
        if sampled_action_length != 0:
            action = full_trajectory[start_idx:(start_idx + sampled_action_length + 1)]
            action = to_relative_frame(action)
            action = action[1:]
            if action.shape[0] < self._action_length:
                last_action = action[-1]
                last_action = np.tile(last_action, (self._action_length - action.shape[0], 1))
                action = np.concatenate((action, last_action), axis=0)
            action = tuple([(e[0], e[1], e[2]) for e in action])
        else:
            action = tuple([(0., 0., 0.) for _ in range(self._action_length)])

        sample = (context, observation, negative_goal, goal, goal_distance, action)

        return sample

    def _sample_negative_goal_image(self, all_trajectories: list[Path]) -> Path:
        trajectory = np.random.choice(all_trajectories)
        images = sorted((trajectory / self._images_dir_name).glob(f"*.{self._image_extension}"))
        image = np.random.choice(images)
        return image

    def _process_error(self, message: str) -> list[tuple[Any]]:
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


class MultiprocessSampleWrapper:

    N_WORKERS_NO_MULTIPROCESS = 0

    def __init__(self,
                 sampler: AbstractEntitySampler,
                 n_workers: int) -> None:
        assert isinstance(n_workers, int) and n_workers >= 0, f"n_workers must be int >= 0, got {n_workers}"
        self._sampler = sampler
        self._n_workers = n_workers

    def __call__(self, input_paths: list[Path]) -> list[Any]:
        process_fn = partial(self._do_parse, all_trajectories=input_paths)

        if len(input_paths) == 1:
            result = [process_fn(input_paths[0])]
        elif self._n_workers == MultiprocessSampleWrapper.N_WORKERS_NO_MULTIPROCESS:
            result = [process_fn(e) for e in input_paths]
        else:
            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                result = executor.map(process_fn, input_paths)
                result = [e for e in result]

        merged_result = []
        for result_subset in result:
            merged_result = merged_result + result_subset
        return merged_result
        
    def _do_parse(self, input_path: Path, all_trajectories: list[Path] | None = None) -> list[Any]:
        result = []
        try:
            result = self._sampler(input_path, all_trajectories)
        except Exception as e:
            print(f"Exception handled when parsing {input_path}")
            print(traceback.format_exc())
        return result
