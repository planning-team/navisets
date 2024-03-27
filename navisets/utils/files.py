from pathlib import Path


_ROS1_BAG_EXTENSION = "bag"
_DIR_PATTERN = "*/"
_ROS1_BAG_PATTERN = "*.bag"
_ROS2_DB_PATTERN = "*.db3"
_ROS2_METADATA_FILE_NAME = "metadata.yaml"
_HDF5_EXTENSION = "hdf5"


def _check_file_extension(file_path: Path, extension: str) -> bool:
    if file_path.is_file():
        return False
    name_parts = file_path.name.split(".")
    if len(name_parts) < 2:
        return False
    extension = name_parts[-1]
    if extension != extension:
        return False
    return True    


def check_ros1_bag(file_path: Path) -> bool:
    return _check_file_extension(file_path, _ROS1_BAG_EXTENSION)


def check_ros2_bag(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    db_files = sorted(dir_path.glob(_ROS2_DB_PATTERN))
    if len(db_files) != 1:
        return False
    db_file = db_files[0]
    if not db_file.is_file():
        return False
    metadata_file = dir_path / _ROS2_METADATA_FILE_NAME
    if not metadata_file.is_file():
        return False
    return True


def check_hdf5_file(file_path: Path) -> bool:
    return _check_file_extension(file_path, _HDF5_EXTENSION)


def locate_rosbags(source_path: Path,
                   ros1_bags: bool = True,
                   ros2_bags: bool = True) -> list[Path]:
    if source_path.is_file():
        if ros1_bags and check_ros1_bag(source_path):
            return [source_path]
        return []
    elif source_path.is_dir():
        if ros2_bags and check_ros2_bag(source_path):
            return [source_path]
        bags = []
        if ros1_bags:
            bags = bags + locate_ros1_bags(source_path)
        if ros2_bags:
            bags = bags + locate_ros2_bags(source_path)
        return bags
    else:
        return []


def locate_ros1_bags(parent_dir: Path) -> list[Path]:
    return [e for e in parent_dir.glob(_ROS1_BAG_PATTERN) if e.is_file()]


def locate_ros2_bags(parent_dir: Path) -> list[Path]:
    return [e for e in parent_dir.glob(_DIR_PATTERN) if check_ros2_bag(e)]


def locate_hdf5_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        if check_hdf5_file(source_path):
            return [source_path]
        return []
    elif source_path.is_dir():
        return sorted(source_path.glob(f"*.{_HDF5_EXTENSION}"))
    else:
        return []
