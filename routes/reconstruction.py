from fastapi import APIRouter
import os
import SimpleITK as sitk
import numpy as np
import pydicom
import json

from config import DATA_ROOT, PLANNING_ROOT
from services.volume_cropper import crop_volume

router = APIRouter()

# =========================================================
# 🔹 HELPER: Load base CT volume
# =========================================================
def load_base_volume(case_id: str):
    case_path = os.path.join(DATA_ROOT, case_id)

    dicom_files = [
        os.path.join(case_path, f)
        for f in os.listdir(case_path)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        raise RuntimeError("No DICOM files found")

    datasets = [pydicom.dcmread(f) for f in dicom_files]
    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    pixel_spacing = tuple(map(float, datasets[0].PixelSpacing))
    slice_thickness = abs(
        float(datasets[1].ImagePositionPatient[2]) -
        float(datasets[0].ImagePositionPatient[2])
    )
    origin = list(map(float, datasets[0].ImagePositionPatient))

    volume = np.stack([
        ds.pixel_array.astype(np.float32) *
        float(getattr(ds, "RescaleSlope", 1.0)) +
        float(getattr(ds, "RescaleIntercept", 0.0))
        for ds in datasets
    ])

    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((pixel_spacing[1], pixel_spacing[0], slice_thickness))
    image.SetOrigin(origin)

    return image


# =========================================================
# 🔹 CT-like kernels
# =========================================================
def apply_kernel(image: sitk.Image, kernel: str):
    kernel = kernel.lower()

    if kernel == "soft":
        return sitk.CurvatureFlow(image, 0.125, 5)

    elif kernel == "abdomen":
        smooth = sitk.CurvatureFlow(image, 0.125, 3)
        return smooth - 0.25 * sitk.Laplacian(smooth)

    elif kernel == "bone":
        return image - 0.7 * sitk.Laplacian(image)

    elif kernel == "lung":
        return image - 0.4 * sitk.Laplacian(image)

    return image


# =========================================================
# 🔹 Resample slice thickness (Z)
# =========================================================
def resample_slice_thickness(image: sitk.Image, new_thickness: float):
    spacing = image.GetSpacing()
    size = image.GetSize()

    new_spacing = (spacing[0], spacing[1], new_thickness)
    new_size = [
        int(size[i] * spacing[i] / new_spacing[i])
        for i in range(3)
    ]

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID()
    )


# =========================================================
# 🔹 Mild Z-axis enhancement (improves MPR)
# =========================================================
def enhance_z_resolution(image: sitk.Image):
    lap = sitk.Laplacian(image)
    return image - 0.15 * lap


# =========================================================
# 🔹 Save axial DICOM series (MPR-correct)
# =========================================================
def save_as_dicom_series(image: sitk.Image, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[save_as_dicom_series] Saving to directory: {out_dir}")
    print(f"[save_as_dicom_series] Image depth: {image.GetDepth()}")

    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    saved_count = 0
    for z in range(image.GetDepth()):
        try:
            slice_img = sitk.Cast(image[:, :, z], sitk.sitkInt16)

            slice_img.SetMetaData("0008|0060", "CT")
            slice_img.SetMetaData("0020|0013", str(z + 1))
            slice_img.SetMetaData("0028|0030", f"{spacing[0]}\\{spacing[1]}")
            slice_img.SetMetaData("0018|0050", str(spacing[2]))

            # Axial orientation
            slice_img.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")

            # Correct spatial position
            ipp_z = origin[2] + z * spacing[2]
            slice_img.SetMetaData(
                "0020|0032",
                f"{origin[0]}\\{origin[1]}\\{ipp_z}"
            )

            output_file = os.path.join(out_dir, f"IM_{z:04d}.dcm")
            writer.SetFileName(output_file)
            writer.Execute(slice_img)
            saved_count += 1
        except Exception as e:
            print(f"[save_as_dicom_series] ERROR saving slice {z}: {e}")
            raise
    
    print(f"[save_as_dicom_series] Successfully saved {saved_count} DICOM files")


# =========================================================
# 🔥 API: APPLY RECONSTRUCTION
# =========================================================
@router.post("/apply")
def apply_reconstruction(payload: dict):

    case_id = payload["case_id"]
    thickness = payload["slice_thickness_mm"]
    kernel = payload.get("kernel", "soft")
    use_planning = payload.get("use_planning", True)  # Healthcare CT: Use planned volume by default
    planning_data = payload.get("planning")  # Optional: can be passed or loaded from file

    # Replace "/" with "_" for planning file name to avoid path issues
    safe_case_id = case_id.replace("/", "_")
    plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

    # Load planning if not provided and use_planning is True
    if use_planning and not planning_data and os.path.exists(plan_path):
        with open(plan_path, "r") as f:
            planning_data = json.load(f)

    recon_id = f"ST_{thickness}_K_{kernel}"
    recon_dir = os.path.join(PLANNING_ROOT, safe_case_id, "recon")
    os.makedirs(recon_dir, exist_ok=True)

    output_path = os.path.join(recon_dir, f"{recon_id}.nii")

    if os.path.exists(output_path):
        return {
            "type": "reconstruction",
            "case_id": case_id,
            "recon_id": recon_id,
            "volume_url": output_path,
            "dicom_series_url": f"/storage/plans/{safe_case_id}/recon/{recon_id}/",
            "kernel": kernel,
            "slice_thickness_mm": thickness,
            "planning_ref": planning_data,
            "used_planning": use_planning and planning_data is not None
        }

    # ---- Load & reconstruct ----
    image = load_base_volume(case_id)

    # ✅ HEALTHCARE CT: Apply planning FIRST (crop to scanned region - Z range only, no FOV)
    if use_planning and planning_data:
        # Convert SimpleITK to numpy for cropping
        volume_np = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        
        # Crop volume using planning (Z range only, FOV cropping removed)
        cropped_np, _ = crop_volume(
            volume_np,
            planning_data["slice_start"],
            planning_data["slice_end"],
            planning_data.get("fov")  # Pass but won't be used
        )
        
        # Convert back to SimpleITK
        # Get original spacing and origin
        original_spacing = image.GetSpacing()
        original_origin = image.GetOrigin()
        
        # Adjust origin for Z crop only (FOV crop removed - keep original X, Y origin)
        z_positions = []
        case_path = os.path.join(DATA_ROOT, case_id)
        dicom_files = [
            os.path.join(case_path, f)
            for f in os.listdir(case_path)
            if f.lower().endswith(".dcm")
        ]
        if dicom_files:
            datasets = [pydicom.dcmread(f) for f in dicom_files]
            datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
            z_positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]
        
        if z_positions and planning_data["slice_start"] < len(z_positions):
            new_origin_z = z_positions[planning_data["slice_start"]]
        else:
            new_origin_z = original_origin[2] + planning_data["slice_start"] * original_spacing[2]
        
        # Keep original X, Y origin (no FOV crop)
        image = sitk.GetImageFromArray(cropped_np)
        image.SetSpacing(original_spacing)
        image.SetOrigin((original_origin[0], original_origin[1], new_origin_z))

    # ✅ Kernel FIRST (critical) - Enable kernel application
    image = apply_kernel(image, kernel)

    # ✅ Then resample
    image = resample_slice_thickness(image, thickness)

    # ✅ Improve MPR quality
    image = enhance_z_resolution(image)

    sitk.WriteImage(image, output_path)

    dicom_out_dir = os.path.join(recon_dir, recon_id)
    print(f"[Reconstruction] Saving DICOM series to: {dicom_out_dir}")
    save_as_dicom_series(image, dicom_out_dir)
    
    # Verify DICOM files were created
    dicom_files = [f for f in os.listdir(dicom_out_dir) if f.lower().endswith(".dcm")]
    print(f"[Reconstruction] Created {len(dicom_files)} DICOM files")
    if not dicom_files:
        raise RuntimeError(f"Failed to create DICOM files in {dicom_out_dir}")

    return {
        "type": "reconstruction",
        "case_id": case_id,
        "recon_id": recon_id,
        "volume_url": output_path,
        "dicom_series_url": f"/storage/plans/{safe_case_id}/recon/{recon_id}/",
        "kernel": kernel,
        "slice_thickness_mm": thickness,
        "planning_ref": planning_data,
        "used_planning": use_planning and planning_data is not None
    }
