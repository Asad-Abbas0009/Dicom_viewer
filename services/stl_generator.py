"""
STL Mesh Generation Service
Generates 3D STL meshes from DICOM CT volumes using VTK and SimpleITK.
"""

import os
import tempfile
import atexit
import SimpleITK as sitk
import vtk
from typing import Optional
from vtk.util import numpy_support

# Track temp files for cleanup
_temp_files = set()

def cleanup_temp_files():
    """Cleanup all temporary files on exit"""
    for tmp_file in _temp_files:
        try:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
        except Exception as e:
            print(f"[STLGenerator] Error cleaning up temp file {tmp_file}: {e}")

atexit.register(cleanup_temp_files)


def load_dicom_as_vtk(dicom_dir: str) -> vtk.vtkImageData:
    """
    Load DICOM series and convert to VTK ImageData.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        
    Returns:
        vtk.vtkImageData object
        
    Raises:
        ValueError: If DICOM directory is invalid or no series found
        RuntimeError: If DICOM reading fails
    """
    if not os.path.isdir(dicom_dir):
        raise ValueError(f"Invalid DICOM directory: {dicom_dir}")
    
    # Security check: ensure path is absolute and exists
    dicom_dir = os.path.abspath(dicom_dir)
    if not os.path.exists(dicom_dir):
        raise ValueError(f"DICOM directory does not exist: {dicom_dir}")

    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        if not series_ids:
            raise ValueError(f"No DICOM series found in directory: {dicom_dir}")

        files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
        if not files:
            raise ValueError(f"No DICOM files found in series: {series_ids[0]}")
            
        reader.SetFileNames(files)
        image = reader.Execute()
        
        # Get volume array (z, y, x)
        volume = sitk.GetArrayFromImage(image)
        
        # Ensure volume is in HU units (apply rescale if needed)
        # Note: SimpleITK reader should handle this, but verify
        # Check if rescale metadata exists before accessing
        try:
            if hasattr(image, 'GetMetaData'):
                # Try to get RescaleIntercept
                rescale_intercept = 0.0
                rescale_slope = 1.0
                
                try:
                    if image.HasMetaDataKey('0028|1052'):  # RescaleIntercept
                        rescale_intercept = float(image.GetMetaData('0028|1052'))
                except:
                    pass  # Use default 0.0
                
                try:
                    if image.HasMetaDataKey('0028|1053'):  # RescaleSlope
                        rescale_slope = float(image.GetMetaData('0028|1053'))
                except:
                    pass  # Use default 1.0
                
                # Apply rescale only if slope is not 1.0 or intercept is not 0.0
                if rescale_slope != 1.0 or rescale_intercept != 0.0:
                    volume = volume * rescale_slope + rescale_intercept
        except Exception as e:
            print(f"[STLGenerator] Warning: Could not apply rescale parameters: {e}")
            print(f"[STLGenerator] Using volume as-is (assuming already in HU units)")

        # Create VTK image
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
        
        # Set spacing and origin (reverse order for VTK: x, y, z)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        vtk_image.SetSpacing(spacing[2], spacing[1], spacing[0])
        vtk_image.SetOrigin(origin[2], origin[1], origin[0])

        # Convert numpy array to VTK
        # Use FLOAT to handle full HU range (-1000 to +3000)
        # Note: VTK_SHORT would be more memory efficient but may clip extreme HU values
        vtk_array = numpy_support.numpy_to_vtk(
            num_array=volume.ravel(order="C"),
            deep=True,
            array_type=vtk.VTK_FLOAT,  # Use FLOAT for full HU range
        )

        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image
        
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM series: {str(e)}") from e


def generate_stl(
    vtk_image: vtk.vtkImageData, 
    hu: int, 
    name: str,
    smooth: bool = True,
    smoothing_iterations: int = 8
) -> str:
    """
    Generate STL file from VTK image using isosurface extraction.
    Optimized for browser loading with aggressive decimation.
    
    Args:
        vtk_image: VTK ImageData object
        hu: Hounsfield Unit threshold for isosurface
        name: Name identifier for the STL file
        smooth: Whether to apply smoothing (recommended for skin, optional for bone)
        smoothing_iterations: Number of smoothing iterations (default: 8 for performance)
        
    Returns:
        Path to generated STL file
        
    Raises:
        ValueError: If no surface found at threshold
        RuntimeError: If STL generation fails
    """
    try:
        # -----------------------------------------------------
        # 1️⃣ Fast surface extraction
        # -----------------------------------------------------
        extractor = vtk.vtkFlyingEdges3D()
        extractor.SetInputData(vtk_image)
        extractor.SetValue(0, float(hu))
        extractor.Update()

        polydata = extractor.GetOutput()
        
        # Check if polydata has any points
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError(f"No surface found at HU threshold {hu}. Try adjusting the threshold.")

        # -----------------------------------------------------
        # 2️⃣ CRITICAL: DECIMATION (prevents browser crash)
        # -----------------------------------------------------
        print(f"[STLGenerator] Original mesh: {polydata.GetNumberOfPoints()} points")
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(polydata)

        # Balanced decimation - less aggressive for better quality
        # Skin needs more reduction than bone, but not too much
        if hu < 0:        # skin
            decimator.SetTargetReduction(0.75)   # 75% reduction (was 92%)
            print(f"[STLGenerator] Applying 75% decimation for skin (HU < 0)")
        else:             # bone
            decimator.SetTargetReduction(0.50)   # 50% reduction (was 70%)
            print(f"[STLGenerator] Applying 50% decimation for bone (HU >= 0)")

        decimator.PreserveTopologyOn()
        decimator.Update()

        polydata = decimator.GetOutput()
        print(f"[STLGenerator] After decimation: {polydata.GetNumberOfPoints()} points")

        # -----------------------------------------------------
        # 3️⃣ Smoothing (more iterations for better quality)
        # -----------------------------------------------------
        if smooth:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(polydata)
            # Increase iterations for better surface quality
            smoother.SetNumberOfIterations(smoothing_iterations if smoothing_iterations > 10 else 15)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.NonManifoldSmoothingOff()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            polydata = smoother.GetOutput()

        # -----------------------------------------------------
        # 4️⃣ Cleanup duplicate points
        # -----------------------------------------------------
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()

        # -----------------------------------------------------
        # 5️⃣ Normals (better shading + stable STL)
        # -----------------------------------------------------
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(cleaner.GetOutput())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()

        polydata = normals.GetOutput()
        print(f"[STLGenerator] Final mesh: {polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} cells")

        # -----------------------------------------------------
        # 6️⃣ Write STL
        # -----------------------------------------------------
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{name}.stl")
        tmp.close()
        _temp_files.add(tmp.name)

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(tmp.name)
        writer.SetInputData(polydata)
        writer.Write()

        # Get file size for logging
        file_size = os.path.getsize(tmp.name)
        print(f"[STLGenerator] STL file size: {file_size / 1024 / 1024:.2f} MB")

        return tmp.name
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate STL: {str(e)}") from e


def cleanup_stl_file(stl_path: str):
    """
    Manually cleanup a specific STL file.
    
    Args:
        stl_path: Path to STL file to delete
    """
    try:
        if os.path.exists(stl_path) and stl_path in _temp_files:
            os.unlink(stl_path)
            _temp_files.discard(stl_path)
    except Exception as e:
        print(f"[STLGenerator] Error cleaning up STL file {stl_path}: {e}")
