from typing import Literal
import numpy as np
from utils.hu_utils import apply_window
from utils.image_utils import array_to_base64_png


def generate_mpr_slice(
    volume: np.ndarray,
    orientation: Literal["axial", "sagittal", "coronal"],
    slice_index: int,
    window_center: float,
    window_width: float
) -> str:
    """
    Generate a single MPR slice from cropped volume.
    
    Parameters
    ----------
    volume : np.ndarray
        Cropped CT volume (Z, Y, X) in HU
    orientation : str
        "axial", "sagittal", or "coronal"
    slice_index : int
        Slice index along the orientation axis
    window_center : float
        Window center for display
    window_width : float
        Window width for display
    
    Returns
    -------
    str
        Base64 PNG string
    """
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (Z, Y, X)")
    
    z_size, y_size, x_size = volume.shape
    
    if orientation == "axial":
        # Axial: volume[z, :, :]
        if slice_index < 0 or slice_index >= z_size:
            raise ValueError(f"Axial slice index {slice_index} out of range [0, {z_size-1}]")
        slice_img = volume[slice_index, :, :]
    
    elif orientation == "sagittal":
        # Sagittal: volume[:, y, :] (left-right view)
        if slice_index < 0 or slice_index >= y_size:
            raise ValueError(f"Sagittal slice index {slice_index} out of range [0, {y_size-1}]")
        slice_img = volume[:, slice_index, :]
        # Transpose to match display orientation
        slice_img = np.flipud(slice_img)  # Flip vertically for correct orientation
    
    elif orientation == "coronal":
        # Coronal: volume[:, :, x] (front-back view)
        if slice_index < 0 or slice_index >= x_size:
            raise ValueError(f"Coronal slice index {slice_index} out of range [0, {x_size-1}]")
        slice_img = volume[:, :, slice_index]
        # Transpose to match display orientation
        slice_img = np.flipud(slice_img)  # Flip vertically for correct orientation
    
    else:
        raise ValueError(f"Invalid orientation: {orientation}")
    
    # Apply windowing
    windowed = apply_window(slice_img, window_center, window_width)
    
    # Convert to base64 PNG
    return array_to_base64_png(windowed)


def get_mpr_metadata(volume: np.ndarray) -> dict:
    """
    Get MPR metadata (number of slices for each orientation).
    
    Parameters
    ----------
    volume : np.ndarray
        Cropped CT volume (Z, Y, X)
    
    Returns
    -------
    dict
        {
            "axial_slices": int,
            "sagittal_slices": int,
            "coronal_slices": int,
            "volume_shape": [z, y, x]
        }
    """
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D (Z, Y, X)")
    
    z_size, y_size, x_size = volume.shape
    
    return {
        "axial_slices": z_size,
        "sagittal_slices": y_size,
        "coronal_slices": x_size,
        "volume_shape": [z_size, y_size, x_size]
    }
