from typing import Dict, Tuple
import numpy as np


def crop_volume_z(
    volume: np.ndarray,
    slice_start: int,
    slice_end: int
) -> np.ndarray:
    """
    Crop CT volume along Z-axis (slice range).

    Parameters
    ----------
    volume : np.ndarray
        Full CT volume (Z, Y, X)
    slice_start : int
        Starting slice index
    slice_end : int
        Ending slice index (inclusive)

    Returns
    -------
    np.ndarray
        Z-cropped volume
    """

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (Z, Y, X)")

    z_max = volume.shape[0] - 1

    slice_start = max(0, slice_start)
    slice_end = min(z_max, slice_end)

    if slice_start > slice_end:
        raise ValueError("slice_start must be <= slice_end")

    return volume[slice_start:slice_end + 1, :, :]


def crop_volume_xy(
    volume: np.ndarray,
    fov: Dict[str, int]
) -> np.ndarray:
    """
    Crop CT volume in X/Y plane (FOV).

    Parameters
    ----------
    volume : np.ndarray
        CT volume (Z, Y, X)
    fov : Dict[str, int]
        {
          "x_min": int,
          "x_max": int,
          "y_min": int,
          "y_max": int
        }

    Returns
    -------
    np.ndarray
        XY-cropped volume
    """

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (Z, Y, X)")

    _, height, width = volume.shape

    x_min = max(0, fov["x_min"])
    x_max = min(width - 1, fov["x_max"])
    y_min = max(0, fov["y_min"])
    y_max = min(height - 1, fov["y_max"])

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid FOV crop range")

    return volume[:, y_min:y_max + 1, x_min:x_max + 1]


def crop_volume(
    volume: np.ndarray,
    slice_start: int,
    slice_end: int,
    fov: Dict[str, int] = None  # FOV parameter kept for backward compatibility but ignored
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply planning crop: Z range only (FOV cropping removed).
    
    Only slice range (Z-axis) cropping is applied. FOV parameter is accepted
    for backward compatibility but is not used.

    Parameters
    ----------
    volume : np.ndarray
        Full CT volume (Z, Y, X)
    slice_start : int
        Starting slice index
    slice_end : int
        Ending slice index (inclusive)
    fov : Dict[str, int], optional
        Ignored - kept for backward compatibility

    Returns
    -------
    cropped_volume : np.ndarray
        Z-cropped CT volume (full X/Y dimensions preserved)
    updated_geometry : Dict[str, int]
        Geometry metadata after cropping
    """

    # Z crop only (FOV cropping removed)
    z_cropped = crop_volume_z(volume, slice_start, slice_end)

    # Don't apply XY crop - return full X/Y dimensions
    final_volume = z_cropped

    updated_geometry = {
        "num_slices": final_volume.shape[0],
        "height": final_volume.shape[1],
        "width": final_volume.shape[2],
        "slice_start": slice_start,
        "slice_end": slice_end,
        "fov": fov if fov else {}  # Keep for compatibility but don't use
    }

    return final_volume, updated_geometry
