from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
from services.study_manager import get_study
from services.dicom_metadata import extract_series_metadata
from services.cine_generator import generate_cine_frames, generate_single_frame
from services.volume_cropper import crop_volume
from config import DATA_ROOT, PLANNING_ROOT
import os
import json
import numpy as np
from functools import lru_cache
import hashlib

router = APIRouter()

# In-memory cache for volumes (to avoid reloading DICOM for each window change)
_volume_cache: Dict[str, Any] = {}

def get_cached_volume(case_id: str, use_planning: bool, recon_id: Optional[str] = None):
    """Get volume from cache or load it"""
    cache_key = f"{case_id}_{use_planning}_{recon_id or 'none'}"
    
    if cache_key in _volume_cache:
        return _volume_cache[cache_key]
    
    # Load volume (same logic as before)
    if recon_id:
        safe_case_id = case_id.replace("/", "_")
        recon_dir = os.path.join(PLANNING_ROOT, safe_case_id, "recon", recon_id)
        
        if not os.path.exists(recon_dir):
            raise HTTPException(status_code=404, detail=f"Reconstruction not found: {recon_id}")
        
        import pydicom
        dicom_files = [
            os.path.join(recon_dir, f)
            for f in os.listdir(recon_dir)
            if f.lower().endswith(".dcm")
        ]
        
        if not dicom_files:
            raise HTTPException(status_code=404, detail=f"No DICOM files found in reconstruction: {recon_id}")
        
        dicom_files.sort()
        
        slices = []
        for dicom_file in dicom_files:
                ds = pydicom.dcmread(dicom_file)
                pixel_array = ds.pixel_array.astype(np.float32)
                if hasattr(ds, "RescaleSlope"):
                    pixel_array = pixel_array * float(ds.RescaleSlope) + float(getattr(ds, "RescaleIntercept", 0.0))
                slices.append(pixel_array)
        
        volume = np.stack(slices)
        
        ds = pydicom.dcmread(dicom_files[0])
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, (list, tuple)) else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, (list, tuple)) else float(ds.WindowWidth[0])
        else:
            wc, ww = 40, 400
    else:
        case_path = os.path.join(DATA_ROOT, case_id)
        safe_case_id = case_id.replace("/", "_")
        plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

        dicom_data = get_study(case_id, case_path)
        meta = extract_series_metadata(dicom_data)

        if use_planning and os.path.exists(plan_path):
            with open(plan_path) as f:
                planning = json.load(f)
            cropped_volume, _ = crop_volume(
                dicom_data["volume"],
                planning["slice_start"],
                planning["slice_end"],
                planning["fov"]
            )
            volume = cropped_volume
        else:
            volume = dicom_data["volume"]
        
        # Always set window center/width from metadata (for both planning and non-planning cases)
        wc = meta["window"]["center"]
        ww = meta["window"]["width"]
    
    # Store in cache
    _volume_cache[cache_key] = {
        "volume": volume,
        "wc": wc,
        "ww": ww
    }
    
    # Keep cache size manageable (max 3 volumes)
    if len(_volume_cache) > 3:
        oldest_key = list(_volume_cache.keys())[0]
        del _volume_cache[oldest_key]
    
    return _volume_cache[cache_key]


@router.get("/slice/{case_id:path}")
def get_single_slice(
    case_id: str,
    slice_index: int = Query(0, description="Slice index to render"),
    use_planning: bool = Query(True),
    recon_id: Optional[str] = Query(None),
    window_center: Optional[float] = Query(None),
    window_width: Optional[float] = Query(None)
):
    """
    Get a single slice with windowing - for instant preview.
    Much faster than generating all frames.
    """
    cached = get_cached_volume(case_id, use_planning, recon_id)
    volume = cached["volume"]
    
    final_wc = window_center if window_center is not None else cached["wc"]
    final_ww = window_width if window_width is not None else cached["ww"]
    
    # DEBUG: Log windowing parameters and volume stats
    slice_data = volume[slice_index]
    print(f"[Slice] WC={final_wc}, WW={final_ww}")
    print(f"[Slice] Volume HU range: min={volume.min():.1f}, max={volume.max():.1f}")
    print(f"[Slice] Slice {slice_index} HU range: min={slice_data.min():.1f}, max={slice_data.max():.1f}")
    
    # Clamp slice index
    slice_index = max(0, min(slice_index, volume.shape[0] - 1))
    
    # Generate single frame
    frame = generate_single_frame(volume[slice_index], final_wc, final_ww)
    
    return {
        "frame": frame,
        "slice_index": slice_index,
        "total_slices": volume.shape[0],
        "window_center": final_wc,
        "window_width": final_ww
    }


@router.get("/{case_id:path}")
def get_cine(
    case_id: str, 
    use_planning: bool = Query(True, description="Use planning if available"),
    recon_id: Optional[str] = Query(None, description="Reconstruction ID to use"),
    window_center: Optional[float] = Query(None, description="Window center override for display"),
    window_width: Optional[float] = Query(None, description="Window width override for display")
):
    """
    Get cine frames (uses cached volume for faster window changes).
    """
    # Use cached volume loader
    cached = get_cached_volume(case_id, use_planning, recon_id)
    volume = cached["volume"]
    
    # Use override window/level if provided, otherwise use metadata defaults
    final_wc = window_center if window_center is not None else cached["wc"]
    final_ww = window_width if window_width is not None else cached["ww"]
    
    print(f"[Cine] Using window center={final_wc}, width={final_ww} (cached volume)")

    cine = generate_cine_frames(
        volume=volume,
        window_center=final_wc,
        window_width=final_ww,
        fps=10
    )

    return cine
