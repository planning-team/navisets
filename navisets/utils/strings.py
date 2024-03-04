def calculate_zeros_pad(data: list) -> int:
    return len(str(len(data))) + 1


def zfill_zeros_pad(idx: int, n_zeros: int) -> str:
    return str(idx).zfill(n_zeros)
