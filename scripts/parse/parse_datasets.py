import fire

from pathlib import Path
from navisets.utils.files import locate_rosbags, locate_hdf5_files
from navisets.utils.parsers import (CameraReferencedRosbagParser, MulitprocessParseWrapper, OverwritePolicy, 
                                    TopicNotFoundPolicy, HDF5FileParser)


_DEFAULT_RATE = 4.

_DATASET_RELLIS3D = "rellis3d"
_DATASET_HURON = "huron"
_DATASET_MUSOHU = "musohu"
_DATASET_SCAND = "scand_loam"
_DATASET_RECON = "recon"

_IMAGE_TOPICS = {
    _DATASET_RELLIS3D: ("/nerian_right/image_color", "/nerian_stereo/right_image"),
    _DATASET_HURON: ("/fisheye_image/compressed",),
    _DATASET_MUSOHU: ("/zed2/zed_node/rgb/image_rect_color/compressed",),
    _DATASET_SCAND: ("/camera",)
}

_ODOMETRY_TOPICS = {
    _DATASET_RELLIS3D: ("/odometry/filtered",),
    _DATASET_HURON: ("/odometry",),
    _DATASET_MUSOHU: ("/zed2/zed_node/odom",),
    _DATASET_SCAND: ("/integrated_to_init",)
}


def main(src_path: str,
         output_dir: str,
         dataset: str,
         rate: float = _DEFAULT_RATE,
         n_workers: int = MulitprocessParseWrapper.N_WORKERS_NO_MULTIPROCESS,
         overwrite: bool = False):
    supported_datasets = (_DATASET_RELLIS3D, _DATASET_HURON, _DATASET_MUSOHU, _DATASET_SCAND, _DATASET_RECON)
    assert dataset in supported_datasets, f"Unknown dataset option {dataset}, available options are: {supported_datasets}"
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    overwrite = OverwritePolicy.DELETE if overwrite else OverwritePolicy.STOP_SILENT

    if dataset != _DATASET_RECON:
        ros1 = dataset in (_DATASET_RELLIS3D, _DATASET_HURON, _DATASET_MUSOHU, _DATASET_SCAND)
        ros2 = False
        convert_color = dataset in (_DATASET_RELLIS3D, _DATASET_HURON)

        input_paths = locate_rosbags(src_path,
                                    ros1_bags=ros1,
                                    ros2_bags=ros2)
        if len(input_paths) == 0:
            print(f"Source path {src_path} is neither a ROS1 bag or directory with ROS1 bags, skipping")
            return
        print(f"Found {len(input_paths)} rosbags, starting processing")
        
        parser = CameraReferencedRosbagParser(rate_hz=rate,
                                              image_topic_candidates=_IMAGE_TOPICS[dataset],
                                              odometry_topic_candidates=_ODOMETRY_TOPICS[dataset],
                                              convert_color=convert_color,
                                              overwrite=overwrite,
                                              topic_not_found=TopicNotFoundPolicy.IGNORE_LOG)
        
    else:
        input_paths = locate_hdf5_files(src_path)
        if len(input_paths) == 0:
            print(f"Source path {src_path} is doesn't have HDF5 files, skipping")
            return
        print(f"Found {len(input_paths)} HDF5 files, starting processing")
        parser = HDF5FileParser(overwrite=overwrite)

    parser = MulitprocessParseWrapper(parser=parser, n_workers=n_workers)
    
    result = parser(input_paths, output_dir)
    n_extracted = sum([0 if e is None else 1 for e in result])
    print(f"Extracted {n_extracted} trajectories")


if __name__ == "__main__":
    fire.Fire(main)
