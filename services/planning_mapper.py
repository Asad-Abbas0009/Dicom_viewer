from typing import List, Dict, Tuple


def map_scout_pixels_to_z_mm(
    z_pixel_start: int,
    z_pixel_end: int,
    scout_height_px: int,
    z_min_mm: float,
    z_max_mm: float,
) -> Tuple[float, float]:
    """
    Convert scout vertical pixel selection to patient Z coordinates (mm).

    Parameters
    ----------
    z_pixel_start : int
        Top pixel selected on scout
    z_pixel_end : int
        Bottom pixel selected on scout
    scout_height_px : int
        Height of scout image in pixels
    z_min_mm : float
        Minimum Z position of CT volume
    z_max_mm : float
        Maximum Z position of CT volume

    Returns
    -------
    (z_start_mm, z_end_mm)
    """

    if scout_height_px <= 0:
        raise ValueError("Scout height must be positive")

    # Normalize pixel positions
    z_start_ratio = z_pixel_start / scout_height_px
    z_end_ratio = z_pixel_end / scout_height_px

    # Convert to patient coordinates
    z_start_mm = z_min_mm + z_start_ratio * (z_max_mm - z_min_mm)
    z_end_mm = z_min_mm + z_end_ratio * (z_max_mm - z_min_mm)

    return z_start_mm, z_end_mm


def map_z_mm_to_slice_indices(
    z_start_mm: float,
    z_end_mm: float,
    z_positions: List[float],
) -> Tuple[int, int]:
    """
    Convert patient Z coordinates (mm) to nearest CT slice indices.

    Parameters
    ----------
    z_start_mm : float
    z_end_mm : float
    z_positions : List[float]
        Z position of each CT slice

    Returns
    -------
    (slice_start_idx, slice_end_idx)
    """

    if not z_positions:
        raise ValueError("z_positions list is empty")

    # Ensure ascending order
    z_positions_sorted = sorted(z_positions)

    def find_closest_index(z_mm: float) -> int:
        return min(
            range(len(z_positions_sorted)),
            key=lambda i: abs(z_positions_sorted[i] - z_mm)
        )

    start_idx = find_closest_index(z_start_mm)
    end_idx = find_closest_index(z_end_mm)

    # Ensure correct order
    slice_start = min(start_idx, end_idx)
    slice_end = max(start_idx, end_idx)

    return slice_start, slice_end


def generate_planning_geometry(
    z_pixel_start: int,
    z_pixel_end: int,
    scout_height_px: int,
    z_positions: List[float],
) -> Dict[str, float]:
    """
    High-level helper: scout pixels → Z mm → slice indices.

    Parameters
    ----------
    z_pixel_start : int
    z_pixel_end : int
    scout_height_px : int
    z_positions : List[float]

    Returns
    -------
    Dict with Z range and slice range
    """

    z_min_mm = min(z_positions)
    z_max_mm = max(z_positions)

    z_start_mm, z_end_mm = map_scout_pixels_to_z_mm(
        z_pixel_start,
        z_pixel_end,
        scout_height_px,
        z_min_mm,
        z_max_mm,
    )

    slice_start, slice_end = map_z_mm_to_slice_indices(
        z_start_mm,
        z_end_mm,
        z_positions,
    )

    return {
        "z_start_mm": z_start_mm,
        "z_end_mm": z_end_mm,
        "slice_start": slice_start,
        "slice_end": slice_end,
    }
