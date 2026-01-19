"""
Central Study / Volume Manager
Loads DICOM volume ONCE and reuses it across all routes.
"""

from typing import Dict
from services.dicom_loader import load_dicom_series

# In-memory cache
_STUDY_CACHE: Dict[str, dict] = {}


def get_study(case_id: str, case_path: str) -> dict:
    """
    Get or load CT study volume.
    """
    if case_id not in _STUDY_CACHE:
        print(f"[StudyManager] Loading volume for {case_id}")
        _STUDY_CACHE[case_id] = load_dicom_series(case_path)
    else:
        print(f"[StudyManager] Reusing cached volume for {case_id}")

    return _STUDY_CACHE[case_id]


def clear_study(case_id: str):
    if case_id in _STUDY_CACHE:
        del _STUDY_CACHE[case_id]


