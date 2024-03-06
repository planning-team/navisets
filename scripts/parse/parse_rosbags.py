import fire

from pathlib import Path
from navisets.utils.files import locate_rosbags
from navisets.utils.parsers import CameraReferencedRosbagParser, MulitprocessRosbagParseWrapper, OverwritePolicy


_DEFAULT_RATE = 4.

_DATASET_RELLIS3D = "rellis3d"
_DATASET_HURON = "huron"
_DATASET_MUSOHU = "musohu"

_IMAGE_TOPICS = {
    _DATASET_RELLIS3D: ("/nerian_right/image_color", "/nerian_stereo/right_image"),
    _DATASET_HURON: ("/fisheye_image/compressed",),
    _DATASET_MUSOHU: ("/zed2/zed_node/rgb/image_rect_color/compressed",)
}

_ODOMETRY_TOPICS = {
    _DATASET_RELLIS3D: ("/odometry/filtered",),
    _DATASET_HURON: ("/odometry",),
    _DATASET_MUSOHU: ("/zed2/zed_node/odom",)
}


def main(src_path: str,
         output_dir: str,
         dataset: str,
         rate: float = _DEFAULT_RATE,
         n_workers: int = MulitprocessRosbagParseWrapper.N_WORKERS_NO_MULTIPROCESS,
         overwrite: bool = False):
    supported_datasets = (_DATASET_RELLIS3D, _DATASET_HURON, _DATASET_MUSOHU)
    assert dataset in supported_datasets, f"Unknown dataset option {dataset}, available options are: {supported_datasets}"
    src_path = Path(src_path)
    output_dir = Path(output_dir)

    ros1 = dataset in (_DATASET_RELLIS3D, _DATASET_HURON, _DATASET_MUSOHU)
    ros2 = False
    convert_color = dataset in (_DATASET_RELLIS3D, _DATASET_HURON)

    rosbag_paths = locate_rosbags(src_path,
                                  ros1_bags=ros1,
                                  ros2_bags=ros2)
    if len(rosbag_paths) == 0:
        print(f"Source path {
              src_path} is neither a ROS1 bag or directory with ROS1 bags, skipping")
        return
    print(f"Found {len(rosbag_paths)} rosbags, starting processing")

    overwrite = OverwritePolicy.DELETE if overwrite else OverwritePolicy.STOP_SILENT
    parser = MulitprocessRosbagParseWrapper(parser=CameraReferencedRosbagParser(rate_hz=rate,
                                                                                image_topic_candidates=_IMAGE_TOPICS[dataset],
                                                                                odometry_topic_candidates=_ODOMETRY_TOPICS[dataset],
                                                                                convert_color=convert_color,
                                                                                overwrite=overwrite),
                                            n_workers=n_workers)
    result = parser(rosbag_paths, output_dir)
    n_extracted = sum([0 if e is None else 1 for e in result])
    print(f"Extracted {n_extracted} rosbags")


if __name__ == "__main__":
    fire.Fire(main)
