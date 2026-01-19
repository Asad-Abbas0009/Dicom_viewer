from typing import Tuple


def clamp(value: int, min_val: int, max_val: int) -> int:
    """
    Clamp integer value between min and max.
    """
    return max(min_val, min(value, max_val))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value between 0 and 1.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)
