import numpy as np


def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply CT windowing to HU image.

    Returns uint8 image in range [0, 255]
    """

    if width <= 0:
        raise ValueError("Window width must be > 0")
    
    image = image.astype(np.float32, copy=False)

    lower = center - width / 2
    upper = center + width / 2

    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)

    return (windowed * 255).astype(np.uint8)
