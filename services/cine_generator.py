from typing import Dict, List
import numpy as np

from utils.image_utils import array_to_base64_png
from utils.hu_utils import apply_window


def generate_single_frame(
    slice_img: np.ndarray,
    window_center: float,
    window_width: float
) -> str:
    """
    Generate a single frame from a 2D slice - for instant preview.
    
    Parameters
    ----------
    slice_img : np.ndarray
        2D slice (Y, X) in HU
    window_center : float
        Window center for display
    window_width : float
        Window width for display
    
    Returns
    -------
    str
        Base64 encoded PNG image
    """
    # Apply windowing (HU → grayscale)
    windowed = apply_window(
        slice_img,
        center=window_center,
        width=window_width
    )
    
    # Convert to PNG base64
    return array_to_base64_png(windowed)


def generate_cine_frames(
    volume: np.ndarray,
    window_center: float,
    window_width: float,
    fps: int = 10
) -> Dict[str, List[str]]:
    """
    Generate cine frames from a planned CT volume.

    Parameters
    ----------
    volume : np.ndarray
        Planned CT volume (Z, Y, X) in HU
    window_center : float
        Window center for display
    window_width : float
        Window width for display
    fps : int
        Frames per second for cine playback

    Returns
    -------
    Dict
        {
          "frames": [base64_png, ...],
          "fps": fps,
          "num_frames": int
        }
    """

    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (Z, Y, X)")

    frames: List[str] = []

    for slice_idx in range(volume.shape[0]):
        slice_img = volume[slice_idx, :, :]

        # Apply windowing (HU → grayscale)
        windowed = apply_window(
            slice_img,
            center=window_center,
            width=window_width
        )

        # Convert to PNG base64
        frame_b64 = array_to_base64_png(windowed)
        frames.append(frame_b64)

    return {
        "frames": frames,
        "fps": fps,
        "num_frames": len(frames)
    }
