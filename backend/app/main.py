"""
main.py - FastAPI Application for 3D Lung Nodule Detection and Cancer Grading.

Endpoints:
- POST /api/predict: Predicts nodules, ordinal malignancy, grade, and activation map for a scan.
- POST /api/upload: Accepts a CT image or volume upload and returns a prediction.
- GET /api/patients: Returns the sample patient cohort.
- GET /api/stats: Aggregate metrics and validation benchmark statistics.
- GET /api/health: Service health, model load status, and label source.
- GET /api/report/{id}: Generates a downloadable JSON clinical report.
"""

import os
import sys
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware

# Ensure app modules can be resolved
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for _p in (BACKEND_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.schemas import (
    PredictRequest,
    PredictResponse,
    PatientRecord,
    StatsResponse,
    HealthResponse,
)
from app.predictor import PredictorService
from app.stats import compute_aggregate_stats, resolve_data_file
from app.slice_analysis import analyze_slice

DISCLAIMER = "Research prototype. Not a medical device. Not for clinical diagnosis."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VOLUME_EXTENSIONS = {".mhd", ".raw", ".dcm", ".nii", ".nrrd"}

app = FastAPI(
    title="Lung Nodule Detection & Cancer Grading API",
    description="Offline-capable 3D Deep Learning service for pulmonary nodule triage and cancer grading.",
    version="1.0.0",
)

# CORS: allow configured origins plus local/preview development hosts
_cors_env = os.getenv("CORS_ORIGIN", "")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]
for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
    if origin not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https?://.*",  # local preview proxies and sandbox hosts
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.getenv("DATA_DIR", "./data")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
UPLOAD_DIR = os.path.join(ARTIFACTS_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

predictor_service = PredictorService(artifacts_dir=ARTIFACTS_DIR, data_dir=DATA_DIR)

# Recent predictions so uploaded scans can be reported on later
PREDICTION_CACHE: Dict[str, Dict] = {}


def _cache_prediction(prediction: Dict) -> None:
    seriesuid = prediction.get("seriesuid")
    if seriesuid:
        prediction["generated_at"] = datetime.now(timezone.utc).isoformat()
        PREDICTION_CACHE[seriesuid] = prediction


def _load_patients() -> List[Dict]:
    patient_file = resolve_data_file(DATA_DIR, "sample_patients.json")
    if not patient_file:
        return []
    with open(patient_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Healthcheck endpoint returning system and model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor_service.is_real_model,
        "label_source": predictor_service.label_source,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/patients", response_model=List[PatientRecord])
def get_patients():
    """Returns the cohort of sample patients."""
    patients = _load_patients()
    if not patients:
        raise HTTPException(status_code=404, detail="sample_patients.json not found")
    return patients


@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    """Returns aggregate cohort numbers and model validation metrics."""
    patient_file = resolve_data_file(DATA_DIR, "sample_patients.json") or os.path.join(DATA_DIR, "sample_patients.json")
    return compute_aggregate_stats(patients_file_path=patient_file, artifacts_dir=ARTIFACTS_DIR)


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Executes multi-task 3D nodule detection and ordinal grading."""
    try:
        prediction = predictor_service.predict(
            seriesuid=request.seriesuid,
            patient_meta=request.patient,
        )
        _cache_prediction(prediction)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/upload", response_model=PredictResponse)
async def upload_scan(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
):
    """
    Accepts a CT scan upload (PNG/JPG image or MetaImage .mhd/.raw volume),
    analyzes it, and returns a prediction with an activation-map overlay.
    """
    filename = os.path.basename(file.filename or "scan.png")
    seriesuid = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    patient_meta = {
        "name": patient_name or f"Patient {seriesuid[:8]}",
        "age": patient_age if patient_age is not None else 60,
    }

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    upload_path = os.path.join(UPLOAD_DIR, filename)
    with open(upload_path, "wb") as f_out:
        f_out.write(contents)

    # Sidecar .raw files that accompany an .mhd upload
    if ext == ".raw":
        mhd_candidate = os.path.join(UPLOAD_DIR, f"{seriesuid}.mhd")
        if os.path.exists(mhd_candidate):
            ext = ".mhd"

    if ext in IMAGE_EXTENSIONS:
        try:
            prediction = analyze_slice(contents, patient_meta=patient_meta)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not analyze image: {exc}")
        prediction["seriesuid"] = seriesuid
        prediction["label_source"] = predictor_service.label_source
        _cache_prediction(prediction)
        return prediction

    if ext in VOLUME_EXTENSIONS:
        prediction = predictor_service.predict(
            seriesuid=seriesuid,
            patient_meta=patient_meta,
            mhd_path=upload_path if ext in (".mhd", ".nii", ".nrrd", ".dcm") else None,
        )
        _cache_prediction(prediction)
        return prediction

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{ext or filename}'. Upload a PNG/JPG image or an .mhd volume.",
    )


@app.get("/api/report/{patient_id}")
def download_patient_report(patient_id: str):
    """Generates a downloadable clinical report JSON for a patient or uploaded scan."""
    patients = _load_patients()
    matched_patient = next(
        (p for p in patients if p["id"] == patient_id or p["seriesuid"] == patient_id),
        None,
    )

    if matched_patient is not None:
        seriesuid = matched_patient["seriesuid"]
        prediction = PREDICTION_CACHE.get(seriesuid)
        if prediction is None:
            prediction = predictor_service.predict(
                seriesuid=seriesuid,
                patient_meta=matched_patient,
            )
            _cache_prediction(prediction)
        patient_payload = matched_patient
    else:
        prediction = PREDICTION_CACHE.get(patient_id)
        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail=f"No patient or analyzed scan found for '{patient_id}'. Run a prediction first.",
            )
        patient_payload = prediction.get("patient") or {"id": patient_id}

    report_payload = {
        "report_id": f"REP-{patient_id}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient": patient_payload,
        "findings": prediction,
        "disclaimer": DISCLAIMER,
    }

    return Response(
        content=json.dumps(report_payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{patient_id}.json"'},
    )
