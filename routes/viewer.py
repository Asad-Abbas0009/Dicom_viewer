# from httpcore import Response
from fastapi import APIRouter, Query, HTTPException
from typing import Literal
from services.study_manager import get_study
from services.volume_cropper import crop_volume
from services.cine_generator import generate_cine_frames
from services.dicom_metadata import extract_series_metadata
from services.mpr_generator import generate_mpr_slice, get_mpr_metadata
from utils.image_utils import array_to_base64_png
from config import DATA_ROOT, PLANNING_ROOT
from fastapi.responses import Response

import os
import json
import numpy as np
import base64

router = APIRouter()

# IMPORTANT: More specific routes (like /mpr/*) must be defined BEFORE catch-all routes (like /{case_id:path})
# FastAPI matches routes in order, so specific routes need to come first

@router.get("/mpr/metadata/{case_id:path}")
def get_mpr_metadata_endpoint(case_id: str):
    """
    Get MPR metadata (number of slices for each orientation).
    Always uses original full volume (no planning, no reconstruction).
    """
    print(f"[MPR Metadata] Loading metadata for case: {case_id} (original volume only)")
    
    case_path = os.path.join(DATA_ROOT, case_id)
    print(f"[MPR Metadata] Case path: {case_path}")
    print(f"[MPR Metadata] Case path exists: {os.path.exists(case_path)}")
    
    if not os.path.exists(case_path):
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    
    try:
        dicom_data = get_study(case_id, case_path)
        full_volume = dicom_data["volume"]
        print(f"[MPR Metadata] Full volume shape: {full_volume.shape}")
        
        # Always use full volume for MPR (no planning, no reconstruction)
        metadata = get_mpr_metadata(full_volume)
        print(f"[MPR Metadata] Metadata: {metadata}")
        
        print(f"[MPR Metadata] Returning metadata successfully")
        return metadata
    except FileNotFoundError as e:
        print(f"[MPR Metadata] FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    except Exception as e:
        print(f"[MPR Metadata] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get MPR metadata: {str(e)}")


@router.get("/mpr/slice/{case_id:path}")
def get_mpr_slice_endpoint(
    case_id: str,
    orientation: Literal["axial", "sagittal", "coronal"] = Query(...),
    slice_index: int = Query(...)
):
    """
    Get MPR slice for a specific orientation and slice index.
    Always uses original full volume (no planning, no reconstruction).
    """
    print(f"[MPR Slice] Loading case: {case_id}, orientation: {orientation}, slice: {slice_index} (original volume only)")
    
    case_path = os.path.join(DATA_ROOT, case_id)
    
    if not os.path.exists(case_path):
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    
    try:
        print(f"[MPR Slice] Using original volume (no reconstruction)")
        dicom_data = get_study(case_id, case_path)
        meta = extract_series_metadata(dicom_data)
        full_volume = dicom_data["volume"]
        print(f"[MPR Slice] Full volume shape: {full_volume.shape}")
        
        # Always use full volume for MPR (no planning, no reconstruction)
        print(f"[MPR Slice] Using full volume (no planning)")
        
        # Generate MPR slice
        png_base64 = generate_mpr_slice(
            volume=full_volume,
            orientation=orientation,
            slice_index=slice_index,
            window_center=meta["window"]["center"],
            window_width=meta["window"]["width"]
        )
        
        png_bytes = base64.b64decode(png_base64)
        return Response(content=png_bytes, media_type="image/png")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[MPR Slice] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate MPR slice: {str(e)}")


@router.get("/slice/{case_id:path}/{slice_index}")
def get_viewer_slice(case_id: str, slice_index: int, use_planning: bool = True, recon_id: str = None):
    """
    Get viewer slice.
    - If recon_id is provided: loads from reconstructed DICOM series
    - If use_planning=True and planning exists: returns cropped planned volume
    - If use_planning=False or no planning: returns full volume
    """
    # If reconstruction ID is provided, load from reconstructed DICOM series
    if recon_id:
        safe_case_id = case_id.replace("/", "_")
        recon_dir = os.path.join(PLANNING_ROOT, safe_case_id, "recon", recon_id)
        
        if not os.path.exists(recon_dir):
            raise HTTPException(status_code=404, detail=f"Reconstruction not found: {recon_id}")
        
        # Load reconstructed DICOM series
        dicom_files = [
            os.path.join(recon_dir, f)
            for f in os.listdir(recon_dir)
            if f.lower().endswith(".dcm")
        ]
        
        if not dicom_files:
            raise HTTPException(status_code=404, detail=f"No DICOM files found in reconstruction: {recon_id}")
        
        dicom_files.sort()  # Sort by filename (IM_0000.dcm, IM_0001.dcm, ...)
        
        if slice_index < 0 or slice_index >= len(dicom_files):
            raise HTTPException(status_code=400, detail=f"Slice index {slice_index} out of range [0, {len(dicom_files)-1}]")
        
        # Load the specific slice
        import pydicom
        ds = pydicom.dcmread(dicom_files[slice_index])
        
        # Get pixel array and apply rescale
        pixel_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope"):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(getattr(ds, "RescaleIntercept", 0.0))
        
        slice_img = pixel_array
        
        # Get window/level from DICOM or use defaults
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (list, tuple)) else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (list, tuple)) else float(ds.WindowWidth[0])
        else:
            # Default window/level
            wc = 40
            ww = 400
    else:
        # Original volume logic
        case_path = os.path.join(DATA_ROOT, case_id)
        # Replace "/" with "_" for planning file name to avoid path issues
        safe_case_id = case_id.replace("/", "_")
        plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

        dicom_data = get_study(case_id, case_path)
        meta = extract_series_metadata(dicom_data)

        # Check if planning exists and should be used
        if use_planning and os.path.exists(plan_path):
            with open(plan_path) as f:
                planning = json.load(f)

            cropped_volume, _ = crop_volume(
                dicom_data["volume"],
                planning["slice_start"],
                planning["slice_end"],
                planning["fov"]
            )

            if slice_index < 0 or slice_index >= cropped_volume.shape[0]:
                return {"error": "Slice out of range"}

            slice_img = cropped_volume[slice_index]
            wc = meta["window"]["center"]
            ww = meta["window"]["width"]
        else:
            # Use full volume (Flow 1: Normal scan without planning)
            full_volume = dicom_data["volume"]
            
            if slice_index < 0 or slice_index >= full_volume.shape[0]:
                return {"error": "Slice out of range"}

            slice_img = full_volume[slice_index]
            wc = meta["window"]["center"]
            ww = meta["window"]["width"]

    # Apply window/level (simple, backend-safe)
    low = wc - ww / 2
    high = wc + ww / 2

    slice_img = np.clip(slice_img, low, high)
    slice_img = ((slice_img - low) / ww * 255).astype(np.uint8)

    # Convert to PNG using existing utility
    png_base64 = array_to_base64_png(slice_img)
    png_bytes = base64.b64decode(png_base64)

    return Response(content=png_bytes, media_type="image/png")


def get_tissue_type(hu_value: float) -> str:
    """Classify tissue based on HU value."""
    if hu_value < -950:
        return "Air"
    elif hu_value < -100:
        return "Fat"
    elif hu_value < 50:
        return "Soft Tissue"
    elif hu_value < 300:
        return "Muscle"
    elif hu_value < 700:
        return "Bone"
    else:
        return "Dense Bone"


@router.get("/hu/{case_id:path}/{slice_index}")
def get_hu_value(
    case_id: str,
    slice_index: int,
    x: int = Query(..., description="Pixel X coordinate"),
    y: int = Query(..., description="Pixel Y coordinate")
):
    """
    Get HU (Hounsfield Unit) value at specific pixel location.
    Returns actual HU value and tissue type classification.
    Always uses full volume (planning is not used for HU values).
    """
    case_path = os.path.join(DATA_ROOT, case_id)
    
    if not os.path.exists(case_path):
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    
    try:
        # Load DICOM volume
        dicom_data = get_study(case_id, case_path)
        volume = dicom_data["volume"]  # (Z, Y, X) in HU
        
        # Validate slice index
        if slice_index < 0 or slice_index >= volume.shape[0]:
            raise HTTPException(status_code=400, detail=f"Slice index {slice_index} out of range [0, {volume.shape[0]-1}]")
        
        # Validate pixel coordinates
        if x < 0 or x >= volume.shape[2] or y < 0 or y >= volume.shape[1]:
            raise HTTPException(status_code=400, detail=f"Pixel coordinates ({x}, {y}) out of range [0, {volume.shape[2]-1}] x [0, {volume.shape[1]-1}]")
        
        # Get HU value
        hu_value = float(volume[slice_index, y, x])
        
        # Determine tissue type
        tissue_type = get_tissue_type(hu_value)
        
        return {
            "hu_value": hu_value,
            "pixel_x": x,
            "pixel_y": y,
            "slice_index": slice_index,
            "tissue_type": tissue_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get HU value: {str(e)}")


@router.get("/cine/{case_id:path}")
def get_viewer_cine(case_id: str, use_planning: bool = True):
    """
    Get cine frames for viewer (legacy endpoint, redirects to /api/cine).
    - If use_planning=True and planning exists: returns cropped planned volume (Flow 2)
    - If use_planning=False or no planning: returns full volume (Flow 1)
    """
    case_path = os.path.join(DATA_ROOT, case_id)
    # Replace "/" with "_" for planning file name to avoid path issues
    safe_case_id = case_id.replace("/", "_")
    plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

    dicom_data = get_study(case_id, case_path)
    meta = extract_series_metadata(dicom_data)

    # Check if planning exists and should be used (Flow 2: Planned scan)
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
        # Use full volume (Flow 1: Normal scan without planning)
        volume = dicom_data["volume"]

    cine = generate_cine_frames(
        volume=volume,
        window_center=meta["window"]["center"],
        window_width=meta["window"]["width"],
        fps=10
    )

    return cine
