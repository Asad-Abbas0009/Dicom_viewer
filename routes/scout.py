from fastapi import APIRouter, Query
import os
import json
from typing import Literal

from services.study_manager import get_study
from services.scout_generator import generate_scout
from services.dicom_metadata import extract_series_metadata
from services.scout_summary import build_scout_summary
from utils.image_utils import array_to_base64_png
from config import DATA_ROOT, PLANNING_ROOT

router = APIRouter()

# --------------------------------------------------
# 1️⃣ Scout SUMMARY API (CT console table)
# --------------------------------------------------
@router.get("/summary/{case_id:path}")
def get_scout_summary(
    case_id: str,
    scout_plane: Literal["frontal", "lateral"] = Query("frontal")
):
    case_path = os.path.join(DATA_ROOT, case_id)
    # Replace "/" with "_" for planning file name to avoid path issues
    safe_case_id = case_id.replace("/", "_")
    plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

    dicom_data = get_study(case_id, case_path)

    meta = extract_series_metadata(dicom_data)

    # Default Z range = full scan
    z_start_mm = min(dicom_data["z_positions"])
    z_end_mm = max(dicom_data["z_positions"])

    # If planning exists → use planned range
    if os.path.exists(plan_path):
        with open(plan_path, "r") as f:
            plan = json.load(f)
            z_start_mm = plan["z_start_mm"]
            z_end_mm = plan["z_end_mm"]

    summary = build_scout_summary(
        case_id=case_id,
        scout_plane=scout_plane,
        z_start_mm=z_start_mm,
        z_end_mm=z_end_mm,
        window_center=meta["window"]["center"],
        window_width=meta["window"]["width"],
        kv=120,              # simulated protocol
        ma=30,
        voice="ON",
        light_timer_sec=5,
        scout_num=1
    )

    return {
        "case_id": case_id,
        "scout_summary": summary
    }


# --------------------------------------------------
# 2️⃣ Scout IMAGE API (frontal / lateral)
# --------------------------------------------------
@router.get("/{case_id:path}")
def get_scout(
    case_id: str,
    type: Literal["frontal", "lateral"] = Query("frontal")
):
    case_path = os.path.join(DATA_ROOT, case_id)

    dicom_data = get_study(case_id, case_path)
    volume = dicom_data["volume"]
    z_positions = dicom_data["z_positions"]

    scout_img = generate_scout(volume, scout_type=type)
    scout_b64 = array_to_base64_png(
        (scout_img * 255).astype("uint8")
    )

    return {
        "scout_type": type,
        "scout_image": scout_b64,
        "rows": scout_img.shape[0],
        "cols": scout_img.shape[1],
        "z_min": min(z_positions),
        "z_max": max(z_positions)
    }
