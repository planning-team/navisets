import fire
import gdown

from pathlib import Path


_LINKS = (
    "https://drive.google.com/drive/folders/1IZ-Tn_kzkp82mNbOL_4sNAniunD7tsYU",
    "https://drive.google.com/drive/folders/1hf-vF5zyTKcCLqIiddIGdemzKT742T1t",
    "https://drive.google.com/drive/folders/1R8jP5Qo7Z6uKPoG9XUvFCStwJu6rtliu",
    "https://drive.google.com/drive/folders/1iP0k6dbmPdAH9kkxs6ugi6-JbrkGhm5o",
    "https://drive.google.com/drive/folders/1WV9pecF2beESyM7N29W-nhi-JaoKvEqc",
)


def main(output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for link in _LINKS:
        gdown.download_folder(url=link, output=str(output_dir))


if __name__ == "__main__":
    fire.Fire(main)
