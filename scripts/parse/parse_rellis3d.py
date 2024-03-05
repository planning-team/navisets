import fire

from pathlib import Path
from navisets.utils.files import locate_rosbags
from navisets.utils.parsers import CameraReferencedRosbagParser, MulitprocessRosbagParseWrapper, OverwritePolicy


_DEFAULT_RATE = 4.

_IMAGE_TOPICS = ("/nerian_right/image_color", "/nerian_stereo/right_image")
_ODOMETRY_TOPICS = ("/odometry/filtered",)


def main(src_path: str,
         output_dir: str,
         rate: float = _DEFAULT_RATE,
         n_workers: int = MulitprocessRosbagParseWrapper.N_WORKERS_NO_MULTIPROCESS,
         overwrite: bool = False):
    src_path = Path(src_path)
    output_dir = Path(output_dir)

    rosbag_paths = locate_rosbags(src_path,
                                  ros2_bags=False)
    if len(rosbag_paths) == 0:
        print(f"Source path {
              src_path} is neither a ROS1 bag or directory with ROS1 bags, skipping")
        return
    print(f"Found {len(rosbag_paths)} rosbags, starting processing")

    overwrite = OverwritePolicy.DELETE if overwrite else OverwritePolicy.STOP_SILENT
    parser = MulitprocessRosbagParseWrapper(parser=CameraReferencedRosbagParser(rate_hz=rate,
                                                                                image_topic_candidates=_IMAGE_TOPICS,
                                                                                odometry_topic_candidates=_ODOMETRY_TOPICS,
                                                                                overwrite=overwrite),
                                            n_workers=n_workers)
    result = parser(rosbag_paths, output_dir)
    n_extracted = sum([0 if e is None else 1 for e in result])
    print(f"Extracted {n_extracted} rosbags")


if __name__ == "__main__":
    fire.Fire(main)
