import shutil
import pickle
import json
import fire

from pathlib import Path
from navisets.utils.samplers import ContextActionTrajectorySampler, MultiprocessSampleWrapper
from navisets.constants import DEFAULT_TRAJECTORY_FILE_NAME


def main(src_path: str,
         output_dir: str,
         context_length: int,
         action_length: int,
         goal_offset: tuple[int, int],
         n_discard_first_frames: int = 0,
         n_discard_last_frames: int = 0,
         sample_interval: int = 1,
         n_workers: int = MultiprocessSampleWrapper.N_WORKERS_NO_MULTIPROCESS,
         overwrite: bool = False,
         no_negative: bool = False):
    if not isinstance(goal_offset, int):
        goal_offset = tuple(goal_offset)
        
    src_path = Path(src_path)
    if (src_path / DEFAULT_TRAJECTORY_FILE_NAME).is_file():
        input_dirs = [src_path]
    else:
        input_dirs = sorted(src_path.glob("*/"))

    output_dir = Path(output_dir)
    if output_dir.is_dir():
        if overwrite:
            shutil.rmtree(str(output_dir))
        else:
            print(f"Output dir {output_dir} already exists, skipping")
            return
    output_dir.mkdir(parents=True, exist_ok=False)

    negative_mining = not no_negative
    metadata = {
        "context_length": context_length,
        "action_length": action_length,
        "goal_offset": goal_offset,
        "negative_mining": negative_mining,
        "n_discard_first_frames": n_discard_first_frames,
        "n_discard_last_frames": n_discard_last_frames,
        "sample_interval": sample_interval
    }
    sampler = ContextActionTrajectorySampler(context_length=context_length,
                                             action_length=action_length,
                                             goal_offset=goal_offset,
                                             negative_mining=negative_mining,
                                             n_discard_first_frames=n_discard_first_frames,
                                             n_discard_last_frames=n_discard_last_frames,
                                             sample_interval=sample_interval)
    sampler = MultiprocessSampleWrapper(sampler, n_workers)
    data = sampler(input_dirs)

    with open(str(output_dir / "data.pickle"), "wb") as f:
        pickle.dump(data, f)
    with open(str(output_dir / "info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sampled {len(data)} samples")


if __name__ == "__main__":
    fire.Fire(main)
