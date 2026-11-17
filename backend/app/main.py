"""
main.py - FastAPI Application for 3D Lung Nodule Detection and Cancer Grading.

Endpoints:
- POST /api/predict: Predicts nodules, ordinal malignancy, grade, and Grad-CAM for a scan.
- POST /api/upload: Accepts CT file upload (.mhd/.raw) and returns multi-task prediction.
- GET /api/patients: Returns the 12 sample patients cohort.
- GET /api/stats: Aggregate metrics and validation benchmark statistics.
- GET /api/health: Service health, model load status, and label source.
- GET /api/report/{id}: Generates downloadable JSON clinical report for a patient.
"""

import os
import sys
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware

# Ensure app modules can be resolved
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import (
    PredictRequest,
    PredictResponse,
    PatientRecord,
    StatsResponse,
    HealthResponse,
)
from app.predictor import PredictorService
from app.stats import compute_aggregate_stats

app = FastAPI(
    title="Lung Nodule Detection & Cancer Grading API",
    description="Offline-capable 3D Deep Learning service for pulmonary nodule triage and cancer grading.",
    version="1.0.0",
)

# CORS: strict local origin configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Service
DATA_DIR = os.getenv("DATA_DIR", "./data")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
predictor_service = PredictorService(artifacts_dir=ARTIFACTS_DIR, data_dir=DATA_DIR)


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Healthcheck endpoint returning system and model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor_service.is_real_model,
        "label_source": predictor_service.label_source,
        "disclaimer": "Research prototype. Not a medical device. Not for clinical diagnosis.",
    }


@app.get("/api/patients", response_model=List[PatientRecord])
def get_patients():
    """Returns the cohort of sample patients."""
    patient_file = os.path.join(DATA_DIR, "sample_patients.json")
    if not os.path.exists(patient_file):
        patient_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_patients.json")

    if not os.path.exists(patient_file):
        raise HTTPException(status_code=404, detail="sample_patients.json not found")

    with open(patient_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    """Returns aggregate cohort numbers and model validation metrics."""
    return compute_aggregate_stats(
        patients_file_path=os.path.join(DATA_DIR, "sample_patients.json"),
        artifacts_dir=ARTIFACTS_DIR,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Executes multi-task 3D nodule detection and ordinal grading."""
    try:
        prediction = predictor_service.predict(
            seriesuid=request.seriesuid,
            patient_meta=request.patient,
        )
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
    Accepts CT scan upload (PNG, JPG, JPEG, DICOM, or MetaImage .mhd/.raw)
    and returns multi-task prediction via mock or real model.
    """
    filename = file.filename or "unknown.png"
    seriesuid = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    patient_meta = {
        "name": patient_name or f"Patient {seriesuid[:6]}",
        "age": patient_age or 60,
    }

    # Save the uploaded file temporarily
    upload_dir = os.path.join(ARTIFACTS_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, filename)
    contents = await file.read()
    with open(upload_path, "wb") as f_out:
        f_out.write(contents)

    # For image files (PNG/JPG/JPEG), run mock prediction
    # (a 2D image cannot be fed directly into the 3D model without
    #  volume reconstruction; mock predictor handles this gracefully)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        # Attempt basic image analysis for size estimation
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(contents)).convert("L")
            arr = __import__("numpy").array(img)
            # Estimate "nodule size" from brightest connected region
            threshold = arr.mean() + arr.std()
            bright_pixels = (arr > threshold).sum()
            # Rough diameter estimate assuming 0.5mm/pixel
            estimated_diameter = float(max(2.0, min(30.0, (bright_pixels ** 0.5) * 0.5)))
        except Exception:
            estimated_diameter = 8.0

        prediction = predictor_service.mock_predictor.predict(
            seriesuid=seriesuid,
            patient_meta=patient_meta,
            label_source=predictor_service.label_source,
        )
        return prediction

    # For .mhd/.raw/.dcm files, use the standard predictor pipeline
    prediction = predictor_service.predict(seriesuid=seriesuid, patient_meta=patient_meta)
    return prediction


@app.get("/api/report/{patient_id}")
def download_patient_report(patient_id: str):
    """Generates downloadable clinical report JSON for a given patient."""
    patient_file = os.path.join(DATA_DIR, "sample_patients.json")
    if not os.path.exists(patient_file):
        patient_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_patients.json")

    matched_patient = None
    if os.path.exists(patient_file):
        with open(patient_file, "r", encoding="utf-8") as f:
            patients = json.load(f)
        for p in patients:
            if p["id"] == patient_id or p["seriesuid"] == patient_id:
                matched_patient = p
                break

    if matched_patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    # Run predictor to get complete inference bundle
    prediction = predictor_service.predict(
        seriesuid=matched_patient["seriesuid"],
        patient_meta=matched_patient,
    )

    report_payload = {
        "report_id": f"REP-{patient_id}",
        "generated_at": "2026-11-16T12:00:00Z",
        "patient": matched_patient,
        "ai_findings": prediction,
        "disclaimer": "Research prototype. Not a medical device. Not for clinical diagnosis.",
    }

    report_json = json.dumps(report_payload, indent=2)
    filename = f"report_{patient_id}.json"

    return Response(
        content=report_json,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
