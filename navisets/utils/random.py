import numpy as np


def get_or_sample_int(value: int | tuple[int, int]) -> int:
    if isinstance(value, int):
        return value
    return np.random.randint(value[0], value[1])
