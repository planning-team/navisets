import subprocess

from pathlib import Path


def download_axel(url: str, output_file: Path, n_connections: int):
    p = subprocess.Popen(f"axel -a -n {n_connections} -o {output_file} {url}", shell=True)
    p.wait()
