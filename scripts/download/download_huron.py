import fire
import requests

from html.parser import HTMLParser
from pathlib import Path
from navisets.utils.download import download_axel


_DATASET_HOME = "https://rail.eecs.berkeley.edu/datasets/huron/"

_ENCODING = "utf-8"


class _PageParser(HTMLParser):

    _TAG_IMG = "img"
    _TAG_A = "a"

    _ATTRIBUTE_ALT = "alt"
    _ATTRIBUTE_HREF = "href"

    _LABEL_DIR = "[DIR]"
    _LABEL_BAG = "[   ]"

    _DATASET_LINK = "/datasets/"

    def __init__(self, dir_mode: bool):
        super().__init__()
        self._dir_mode = dir_mode
        self._in_proper_img_tag = False
        self._in_proper_a_tag = False
        self._data = []
        self._current_item = None

    def get_data(self) -> list[str]:
        return self._data.copy()

    def feed(self, data):
        self._data = []
        super().feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == _PageParser._TAG_IMG:
            alt_value = _PageParser._get_attribute_value(attrs, _PageParser._ATTRIBUTE_ALT)
            if self._dir_mode:
                if alt_value == _PageParser._LABEL_DIR:
                    self._in_proper_img_tag = True
            else:
                if alt_value == _PageParser._LABEL_BAG:
                    self._in_proper_img_tag = True

        elif tag == _PageParser._TAG_A and self._in_proper_img_tag:
            link = _PageParser._get_attribute_value(attrs, _PageParser._ATTRIBUTE_HREF)
            if link != _PageParser._DATASET_LINK:
                self._in_proper_a_tag = True
                self._current_item = link

    def handle_endtag(self, tag):
        if tag == "img":
            self._in_proper_img_tag = False
        elif tag == "a":
            self._in_proper_a_tag = False
            self._current_item = None

    def handle_data(self, data):
        if self._in_proper_a_tag:
            self._data.append(self._current_item)

    @staticmethod
    def _get_attribute_value(attributes: tuple[str, ...], target_attribute: str) -> str | None:
        for attribute_name, attribute_value in attributes:
            if attribute_name == target_attribute:
                return attribute_value
        return None


def _download_page(url: str) -> str:
    page = requests.get(url)
    page = page.content.decode(_ENCODING)
    return page


def _download_and_parse_page(url: str, dir_mode: bool) -> list[str]:
    page = _download_page(url)
    parser = _PageParser(dir_mode=dir_mode)
    parser.feed(page)
    return parser.get_data()


def main(output_dir: str, n_axel_connections: int = 1):
    print(f"Parsing dataset home page {_DATASET_HOME}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = _download_and_parse_page(_DATASET_HOME, True)

    bags = []
    for subdir in subdirs:
        subdir_bag_urls = _download_and_parse_page(_DATASET_HOME + subdir, False)
        bags = bags + [(subdir, e) for e in subdir_bag_urls]
    
    print(f"Parsed {len(bags)} rosbag URLs, starting download")
    for subdir, file_name in bags:
        url = _DATASET_HOME + subdir + file_name
        output_file_name = f"{subdir[:-1]}_{file_name}"
        download_axel(url, output_dir / output_file_name, n_axel_connections)

    print("Finished downloading")


if __name__ == "__main__":
    fire.Fire(main)
