"""
STL Mesh Generation Routes
API endpoints for generating 3D STL meshes from DICOM CT volumes.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from services.stl_generator import load_dicom_as_vtk, generate_stl, cleanup_stl_file
from config import DATA_ROOT

router = APIRouter()


# =========================================================
# REQUEST MODELS
# =========================================================
class DICOMRequest(BaseModel):
    """Request model for DICOM directory path"""
    dicom_dir: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "dicom_dir": "Abdomen/CT Abdomen Contrast/case_001"
            }
        }


class STLRequest(BaseModel):
    """Request model for custom STL generation"""
    case_id: str
    hu_threshold: int
    smooth: bool = True
    smoothing_iterations: int = 15
    name: str = "mesh"


# =========================================================
# API: BONE STL DOWNLOAD
# =========================================================
@router.post("/generate/bone")
def generate_bone_stl(req: DICOMRequest, background_tasks: BackgroundTasks):
    """
    Generate bone STL mesh from DICOM series.
    Uses HU threshold of 300 (bone tissue).
    No smoothing applied for sharp bone edges.
    """
    try:
        # Convert case_id path to full directory path
        if "/" in req.dicom_dir or "\\" in req.dicom_dir:
            # It's a case_id path like "Abdomen/CT Abdomen Contrast/case_001"
            dicom_path = os.path.join(DATA_ROOT, req.dicom_dir)
        else:
            # Assume it's a direct path
            dicom_path = req.dicom_dir
        
        if not os.path.exists(dicom_path):
            raise HTTPException(status_code=404, detail=f"DICOM directory not found: {req.dicom_dir}")
        
        vtk_image = load_dicom_as_vtk(dicom_path)
        # Bone: 50% decimation, light smoothing for better quality
        # HU threshold 200 for better bone definition (was 300)
        stl_path = generate_stl(vtk_image, hu=200, name="bone", smooth=True, smoothing_iterations=10)
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_stl_file, stl_path)
        
        return FileResponse(
            stl_path,
            media_type="model/stl",  # Correct MIME type for STL files
            filename="bone.stl"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# =========================================================
# API: SKIN STL DOWNLOAD
# =========================================================
@router.post("/generate/skin")
def generate_skin_stl(req: DICOMRequest, background_tasks: BackgroundTasks):
    """
    Generate skin STL mesh from DICOM series.
    Uses HU threshold of -200 (soft tissue/skin).
    Smoothing is applied for better surface quality.
    """
    try:
        # Convert case_id path to full directory path
        if "/" in req.dicom_dir or "\\" in req.dicom_dir:
            # It's a case_id path like "Abdomen/CT Abdomen Contrast/case_001"
            dicom_path = os.path.join(DATA_ROOT, req.dicom_dir)
        else:
            # Assume it's a direct path
            dicom_path = req.dicom_dir
        
        if not os.path.exists(dicom_path):
            raise HTTPException(status_code=404, detail=f"DICOM directory not found: {req.dicom_dir}")
        
        vtk_image = load_dicom_as_vtk(dicom_path)
        # Skin: 75% decimation + smoothing for better surface quality
        # HU threshold -100 for better skin definition (was -200)
        stl_path = generate_stl(vtk_image, hu=-100, name="skin", smooth=True, smoothing_iterations=15)
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_stl_file, stl_path)
        
        return FileResponse(
            stl_path,
            media_type="model/stl",  # Correct MIME type for STL files
            filename="skin.stl"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# =========================================================
# API: CUSTOM STL GENERATION
# =========================================================
@router.post("/generate/custom")
def generate_custom_stl(req: STLRequest, background_tasks: BackgroundTasks):
    """
    Generate STL mesh with custom HU threshold and settings.
    
    Args:
        case_id: Case identifier (e.g., "Abdomen/CT Abdomen Contrast/case_001")
        hu_threshold: Hounsfield Unit threshold for isosurface extraction
        smooth: Whether to apply smoothing (default: True)
        smoothing_iterations: Number of smoothing iterations (default: 15)
        name: Name for the output file (default: "mesh")
    """
    try:
        # Convert case_id to full directory path
        dicom_path = os.path.join(DATA_ROOT, req.case_id)
        
        if not os.path.exists(dicom_path):
            raise HTTPException(status_code=404, detail=f"Case not found: {req.case_id}")
        
        vtk_image = load_dicom_as_vtk(dicom_path)
        # Custom: Apply decimation based on HU threshold (skin vs bone)
        # Default smoothing_iterations reduced to 8 for better performance
        smoothing_iters = req.smoothing_iterations if req.smoothing_iterations else 8
        stl_path = generate_stl(
            vtk_image, 
            hu=req.hu_threshold, 
            name=req.name,
            smooth=req.smooth,
            smoothing_iterations=smoothing_iters
        )
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_stl_file, stl_path)
        
        return FileResponse(
            stl_path,
            media_type="model/stl",
            filename=f"{req.name}.stl"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# =========================================================
# HEALTH CHECK
# =========================================================
@router.get("/health")
def stl_health_check():
    """Health check endpoint for STL service"""
    return {"status": "STL generation service is running"}
