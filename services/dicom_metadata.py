from typing import Dict, Any, List
import pydicom


def extract_series_metadata(dicom_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CT series metadata required by Cornerstone3D
    and other viewer components.

    Parameters
    ----------
    dicom_data : Dict[str, Any]
        Output from dicom_loader.load_dicom_series()

    Returns
    -------
    Dict[str, Any]
        Viewer-ready metadata
    """

    datasets: List[pydicom.Dataset] = dicom_data["datasets"]
    z_positions: List[float] = dicom_data["z_positions"]

    ref_ds = datasets[0]

    # ------------------------------------------------------------------
    # Basic image geometry
    # ------------------------------------------------------------------
    rows = int(ref_ds.Rows)
    cols = int(ref_ds.Columns)

    pixel_spacing = list(map(float, ref_ds.PixelSpacing))
    slice_thickness = float(dicom_data["slice_thickness"])

    # ------------------------------------------------------------------
    # Orientation & position (CRITICAL for MPR)
    # ------------------------------------------------------------------
    image_orientation = list(map(float, ref_ds.ImageOrientationPatient))
    image_position = list(map(float, ref_ds.ImagePositionPatient))

    # Direction cosines
    row_cosines = image_orientation[:3]
    col_cosines = image_orientation[3:]

    # ------------------------------------------------------------------
    # Windowing (fallback safe)
    # ------------------------------------------------------------------
    def get_window_value(value, default):
        if value is None:
            return default
        if isinstance(value, pydicom.multival.MultiValue):
            return float(value[0])
        return float(value)

    window_center = get_window_value(
        getattr(ref_ds, "WindowCenter", None), default=40
    )

    window_width = get_window_value(
        getattr(ref_ds, "WindowWidth", None), default=400
    )

    # ------------------------------------------------------------------
    # Rescale values (HU)
    # ------------------------------------------------------------------
    rescale_slope = float(getattr(ref_ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ref_ds, "RescaleIntercept", 0.0))

    # ------------------------------------------------------------------
    # Patient & study identifiers (useful later)
    # ------------------------------------------------------------------
    patient_id = getattr(ref_ds, "PatientID", "")
    study_uid = getattr(ref_ds, "StudyInstanceUID", "")
    series_uid = getattr(ref_ds, "SeriesInstanceUID", "")
    modality = getattr(ref_ds, "Modality", "CT")

    # ------------------------------------------------------------------
    # Z geometry
    # ------------------------------------------------------------------
    z_spacing = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else slice_thickness

    # ------------------------------------------------------------------
    # Final metadata dictionary
    # ------------------------------------------------------------------
    metadata = {
        "modality": modality,
        "rows": rows,
        "cols": cols,
        "num_slices": len(datasets),
        "pixel_spacing": pixel_spacing,        # [row, col]
        "slice_thickness": slice_thickness,
        "z_spacing": z_spacing,
        "image_orientation": image_orientation,
        "row_cosines": row_cosines,
        "col_cosines": col_cosines,
        "image_position": image_position,
        "z_positions": z_positions,
        "window": {
            "center": window_center,
            "width": window_width
        },
        "rescale": {
            "slope": rescale_slope,
            "intercept": rescale_intercept
        },
        "uids": {
            "patient_id": patient_id,
            "study_uid": study_uid,
            "series_uid": series_uid
        }
    }

    return metadata
