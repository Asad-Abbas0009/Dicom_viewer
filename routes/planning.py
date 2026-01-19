from http.client import HTTPException
from fastapi import APIRouter
from pydantic import BaseModel
from services.study_manager import get_study
from services.planning_mapper import generate_planning_geometry
from config import DATA_ROOT, PLANNING_ROOT
import os
import json

router = APIRouter()


class PlanningRequest(BaseModel):
    case_id: str
    z_pixel_start: int
    z_pixel_end: int
    scout_height_px: int
    fov: dict


@router.post("/")
def save_planning(req: PlanningRequest):
    case_path = os.path.join(DATA_ROOT, req.case_id)
    dicom_data = get_study(req.case_id, case_path)


    planning = generate_planning_geometry(
        z_pixel_start=req.z_pixel_start,
        z_pixel_end=req.z_pixel_end,
        scout_height_px=req.scout_height_px,
        z_positions=dicom_data["z_positions"]
    )

    planning["fov"] = req.fov

    os.makedirs(PLANNING_ROOT, exist_ok=True)
    # Replace "/" with "_" for planning file name to avoid path issues
    safe_case_id = req.case_id.replace("/", "_")
    plan_path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")

    with open(plan_path, "w") as f:
        json.dump(planning, f, indent=2)

    return {
        "message": "Planning saved",
        "planning": planning
    }

@router.get("/get/{case_id:path}")
def get_planning(case_id: str):
    # Replace "/" with "_" for planning file name to avoid path issues
    safe_case_id = case_id.replace("/", "_")
    path = os.path.join(PLANNING_ROOT, f"{safe_case_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return json.load(open(path))

