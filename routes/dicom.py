from fastapi import APIRouter
from services.study_manager import get_study
from services.dicom_metadata import extract_series_metadata
from config import DATA_ROOT
import os

router = APIRouter()


@router.get("/body-parts")
def list_body_parts():
    """List all available body parts (Level 1)"""
    if not os.path.exists(DATA_ROOT):
        return {"body_parts": []}
    
    body_parts = []
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path):
            body_parts.append(item)
    
    return {"body_parts": sorted(body_parts)}


@router.get("/protocols/{body_part}")
def list_protocols(body_part: str):
    """List all protocols for a given body part (Level 2)"""
    body_part_path = os.path.join(DATA_ROOT, body_part)
    
    if not os.path.exists(body_part_path):
        return {"protocols": []}
    
    protocols = []
    for item in os.listdir(body_part_path):
        item_path = os.path.join(body_part_path, item)
        if os.path.isdir(item_path):
            # Count DICOM files in all cases under this protocol
            dicom_count = 0
            for case_folder in os.listdir(item_path):
                case_path = os.path.join(item_path, case_folder)
                if os.path.isdir(case_path):
                    try:
                        dcm_files = [f for f in os.listdir(case_path) if f.lower().endswith('.dcm')]
                        dicom_count += len(dcm_files)
                    except (PermissionError, OSError):
                        continue
            
            protocols.append({
                "name": item,
                "dicom_count": dicom_count,
                "case_path": f"{body_part}/{item}"
            })
    
    return {"protocols": sorted(protocols, key=lambda x: x["name"])}


@router.get("/cases/{body_part}/{protocol}")
def list_cases(body_part: str, protocol: str):
    """List all cases for a given body part and protocol (Level 3)"""
    protocol_path = os.path.join(DATA_ROOT, body_part, protocol)
    
    if not os.path.exists(protocol_path):
        return {"cases": []}
    
    cases = []
    for item in os.listdir(protocol_path):
        item_path = os.path.join(protocol_path, item)
        if os.path.isdir(item_path):
            try:
                dcm_files = [f for f in os.listdir(item_path) if f.lower().endswith('.dcm')]
                if len(dcm_files) > 0:
                    cases.append({
                        "name": item,
                        "dicom_count": len(dcm_files),
                        "case_path": f"{body_part}/{protocol}/{item}"
                    })
            except (PermissionError, OSError):
                continue
    
    return {"cases": sorted(cases, key=lambda x: x["name"])}


@router.get("/hierarchical-cases")
def list_hierarchical_cases():
    """Get complete hierarchical structure"""
    if not os.path.exists(DATA_ROOT):
        return {"structure": {}}
    
    structure = {}
    
    for body_part in os.listdir(DATA_ROOT):
        body_part_path = os.path.join(DATA_ROOT, body_part)
        if not os.path.isdir(body_part_path):
            continue
        
        protocols = []
        for protocol in os.listdir(body_part_path):
            protocol_path = os.path.join(body_part_path, protocol)
            if not os.path.isdir(protocol_path):
                continue
            
            cases = []
            for case_folder in os.listdir(protocol_path):
                case_path = os.path.join(protocol_path, case_folder)
                if os.path.isdir(case_path):
                    try:
                        dcm_files = [f for f in os.listdir(case_path) if f.lower().endswith('.dcm')]
                        if len(dcm_files) > 0:
                            cases.append({
                                "name": case_folder,
                                "dicom_count": len(dcm_files),
                                "case_path": f"{body_part}/{protocol}/{case_folder}"
                            })
                    except (PermissionError, OSError):
                        continue
            
            if cases:  # Only add protocol if it has cases
                protocols.append({
                    "name": protocol,
                    "dicom_count": sum(c["dicom_count"] for c in cases),
                    "case_path": f"{body_part}/{protocol}",
                    "cases": sorted(cases, key=lambda x: x["name"])
                })
        
        if protocols:  # Only add body part if it has protocols with cases
            structure[body_part] = sorted(protocols, key=lambda x: x["name"])
    
    return {"structure": structure}


@router.get("/metadata/{case_id:path}")
def get_dicom_metadata(case_id: str, recon_id: str = None):
    """
    Get DICOM metadata for a case.
    - If recon_id is provided: loads metadata from reconstructed DICOM series
    - Otherwise: uses original volume metadata
    """
    from config import PLANNING_ROOT
    from fastapi import Query, HTTPException
    
    # If reconstruction ID is provided, load from reconstructed DICOM series
    if recon_id:
        safe_case_id = case_id.replace("/", "_")
        recon_dir = os.path.join(PLANNING_ROOT, safe_case_id, "recon", recon_id)
        
        if not os.path.exists(recon_dir):
            raise HTTPException(status_code=404, detail=f"Reconstruction not found: {recon_id}")
        
        # Load reconstructed DICOM series
        import pydicom
        dicom_files = [
            os.path.join(recon_dir, f)
            for f in os.listdir(recon_dir)
            if f.lower().endswith(".dcm")
        ]
        
        if not dicom_files:
            raise HTTPException(status_code=404, detail=f"No DICOM files found in reconstruction: {recon_id}")
        
        dicom_files.sort()  # Sort by filename
        
        # Load all slices to get metadata
        datasets = []
        z_positions = []
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                datasets.append(ds)
                if hasattr(ds, "ImagePositionPatient"):
                    z_positions.append(float(ds.ImagePositionPatient[2]))
                else:
                    # Fallback: use slice number
                    z_positions.append(float(getattr(ds, "InstanceNumber", len(z_positions))))
            except Exception as e:
                print(f"[DICOM Metadata] Error loading DICOM file {dicom_file}: {e}")
                continue
        
        if not datasets:
            raise HTTPException(status_code=500, detail="Failed to load reconstructed DICOM files")
        
        # Sort by Z position
        if z_positions:
            sorted_indices = sorted(range(len(z_positions)), key=lambda i: z_positions[i])
            datasets = [datasets[i] for i in sorted_indices]
            z_positions = [z_positions[i] for i in sorted_indices]
        
        # Create dicom_data structure similar to load_dicom_series output
        ref_ds = datasets[0]
        rows = int(ref_ds.Rows)
        cols = int(ref_ds.Columns)
        
        # Get slice thickness from DICOM metadata
        slice_thickness = float(getattr(ref_ds, "SliceThickness", 
            abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.0))
        
        # Get pixel spacing
        if hasattr(ref_ds, "PixelSpacing"):
            pixel_spacing = tuple(map(float, ref_ds.PixelSpacing))
        else:
            # Fallback: try to get from metadata string
            try:
                pixel_spacing_str = ref_ds.get("0028|0030", "1.0\\1.0")
                pixel_spacing = tuple(map(float, pixel_spacing_str.split("\\")))
            except:
                pixel_spacing = (1.0, 1.0)
        
        # Get orientation and origin
        if hasattr(ref_ds, "ImageOrientationPatient"):
            orientation = list(map(float, ref_ds.ImageOrientationPatient))
        else:
            orientation = [1, 0, 0, 0, 1, 0]  # Default axial
        
        if hasattr(ref_ds, "ImagePositionPatient"):
            origin = list(map(float, ref_ds.ImagePositionPatient))
        else:
            origin = [0, 0, 0]
        
        dicom_data = {
            "volume": None,  # Not needed for metadata
            "datasets": datasets,
            "z_positions": z_positions,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "orientation": orientation,
            "origin": origin,
            "rows": rows,
            "cols": cols,
        }
        
        metadata = extract_series_metadata(dicom_data)
        return metadata
    else:
        # Original volume logic
        case_path = os.path.join(DATA_ROOT, case_id)
        
        if not os.path.exists(case_path):
            return {"error": f"Case not found: {case_id}"}
        
        dicom_data = get_study(case_id, case_path)
        metadata = extract_series_metadata(dicom_data)
        
        return metadata


@router.get("/files/{case_id:path}")
def list_dicom_files(case_id: str):
    """List DICOM files for a case. case_id can be nested: 'Abdomen/CT Abdomen Contrast/case_001'"""
    case_path = os.path.join(DATA_ROOT, case_id)
    
    if not os.path.exists(case_path):
        return {"error": f"Case not found: {case_id}"}
    
    files = sorted(
        f for f in os.listdir(case_path)
        if f.lower().endswith(".dcm")
    )
    
    return {"files": files}


@router.get("/cases")
def list_cases_legacy():
    """Legacy endpoint - lists all cases in flat structure (for backward compatibility)"""
    if not os.path.exists(DATA_ROOT):
        return {"cases": []}
    
    cases = []
    
    # Walk through hierarchical structure
    for body_part in os.listdir(DATA_ROOT):
        body_part_path = os.path.join(DATA_ROOT, body_part)
        if not os.path.isdir(body_part_path):
            continue
        
        for protocol in os.listdir(body_part_path):
            protocol_path = os.path.join(body_part_path, protocol)
            if not os.path.isdir(protocol_path):
                continue
            
            for case_folder in os.listdir(protocol_path):
                case_path = os.path.join(protocol_path, case_folder)
                if os.path.isdir(case_path):
                    try:
                        dcm_files = [f for f in os.listdir(case_path) if f.lower().endswith('.dcm')]
                        if len(dcm_files) > 0:
                            # Return nested path as case_id
                            cases.append(f"{body_part}/{protocol}/{case_folder}")
                    except (PermissionError, OSError):
                        continue
    
    return {"cases": sorted(cases)}