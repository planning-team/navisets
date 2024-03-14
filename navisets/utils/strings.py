def calculate_zeros_pad(data_length: list) -> int:
    return len(str(data_length)) + 1


def zfill_zeros_pad(idx: int, n_zeros: int) -> str:
    return str(idx).zfill(n_zeros)
