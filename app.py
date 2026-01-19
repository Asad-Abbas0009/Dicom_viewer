# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

# from routes import dicom, scout, planning, cine, viewer

# app = FastAPI(title="CT DICOM Viewer Simulator")

# # CORS (frontend will need this)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173"
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Register routes
# app.include_router(dicom.router, prefix="/api/dicom", tags=["DICOM"])
# app.include_router(scout.router, prefix="/api/scout", tags=["Scout"])
# app.include_router(planning.router, prefix="/api/planning", tags=["Planning"])
# app.include_router(cine.router, prefix="/api/cine", tags=["Cine"])
# app.include_router(viewer.router, prefix="/api/viewer", tags=["Viewer"])

# @app.get("/")
# def health_check():
#     return {"status": "CT backend running"}

# from config import PLANNING_ROOT
# import os
# import shutil

# @app.on_event("startup")
# def clear_old_plans():
#     if os.path.exists(PLANNING_ROOT):
#         shutil.rmtree(PLANNING_ROOT)
#     os.makedirs(PLANNING_ROOT, exist_ok=True)
#     print("✅ Planning state cleared on startup")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import dicom
from routes import scout, planning, cine, viewer, reconstruction, stl
from config import PLANNING_ROOT
import os, shutil

app = FastAPI(title="CT Simulator")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow ALL origins for development
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ROUTES ----------
app.include_router(scout.router, prefix="/api/scout")
app.include_router(planning.router, prefix="/api/planning")
app.include_router(cine.router, prefix="/api/cine")
app.include_router(dicom.router, prefix="/api/dicom")
app.include_router(viewer.router, prefix="/api/viewer")
app.include_router(reconstruction.router, prefix="/api/reconstruct", tags=["Reconstruction"])
app.include_router(stl.router, prefix="/api/stl", tags=["STL"])
# ---------- STATIC DICOM (VERY IMPORTANT) ----------
app.mount(
    "/dicom",
    StaticFiles(directory="data/ct_cases"),
    name="dicom"
)

# ---------- STATIC RECONSTRUCTED DICOM ----------
app.mount(
    "/storage",
    StaticFiles(directory="storage"),
    name="storage"
)



# ---------- HEALTH ----------
@app.get("/")
def health():
    return {"status": "CT backend running"}

# ---------- CLEAN PLANS ON START ----------
@app.on_event("startup")
def clear_plans():
    if os.path.exists(PLANNING_ROOT):
        shutil.rmtree(PLANNING_ROOT)
    os.makedirs(PLANNING_ROOT, exist_ok=True)
