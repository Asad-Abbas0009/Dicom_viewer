import os
from typing import List, Dict, Any

import numpy as np
import pydicom


def load_dicom_series(case_path: str) -> Dict[str, Any]:
    """
    Load a CT DICOM series from a folder and return a correctly
    ordered 3D volume with geometry metadata.

    Parameters
    ----------
    case_path : str
        Path to folder containing DICOM files of one CT series.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing volume, metadata, and geometry.
    """

    if not os.path.isdir(case_path):
        raise FileNotFoundError(f"CT case folder not found: {case_path}")

    # ------------------------------------------------------------------
    # 1️⃣ Read all DICOM files
    # ------------------------------------------------------------------
    dicom_files = [
        os.path.join(case_path, f)
        for f in os.listdir(case_path)
        if f.lower().endswith(".dcm")
    ]

    if len(dicom_files) == 0:
        raise RuntimeError(f"No DICOM files found in {case_path}")

    datasets: List[pydicom.Dataset] = []

    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path)
            datasets.append(ds)
        except Exception:
            # Skip corrupted or unreadable files
            continue

    if len(datasets) == 0:
        raise RuntimeError("No readable DICOM files found")

    # ------------------------------------------------------------------
    # 2️⃣ Sort slices by Z position (ImagePositionPatient)
    # ------------------------------------------------------------------
    def get_z_position(ds: pydicom.Dataset) -> float:
        return float(ds.ImagePositionPatient[2])

    datasets.sort(key=get_z_position)

    z_positions = [get_z_position(ds) for ds in datasets]

    # ------------------------------------------------------------------
    # 3️⃣ Extract geometry metadata (from first slice)
    # ------------------------------------------------------------------
    ref_ds = datasets[0]

    rows = int(ref_ds.Rows)
    cols = int(ref_ds.Columns)

    pixel_spacing = tuple(map(float, ref_ds.PixelSpacing))

    # Slice thickness may be missing or unreliable
    slice_thickness = float(
        getattr(ref_ds, "SliceThickness", abs(z_positions[1] - z_positions[0]))
    )

    orientation = list(map(float, ref_ds.ImageOrientationPatient))
    origin = list(map(float, ref_ds.ImagePositionPatient))

    # Rescale parameters for HU conversion
    rescale_slope = float(getattr(ref_ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ref_ds, "RescaleIntercept", 0.0))

    # ------------------------------------------------------------------
    # 4️⃣ Stack slices into 3D volume (Z, Y, X)
    # ------------------------------------------------------------------
    volume = np.zeros((len(datasets), rows, cols), dtype=np.float32)

    for i, ds in enumerate(datasets):
        pixel_array = ds.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units
        pixel_array = pixel_array * rescale_slope + rescale_intercept

        volume[i, :, :] = pixel_array

    # ------------------------------------------------------------------
    # 5️⃣ Final output
    # ------------------------------------------------------------------
    return {
        "volume": volume,                 # (Z, Y, X)
        "datasets": datasets,              # original DICOM datasets
        "z_positions": z_positions,        # mm
        "pixel_spacing": pixel_spacing,    # (row, col)
        "slice_thickness": slice_thickness,
        "orientation": orientation,
        "origin": origin,
        "rows": rows,
        "cols": cols,
    }
