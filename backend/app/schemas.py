"""
schemas.py - Pydantic Request and Response Schemas.

Strict schemas for API validation across detection, grading, patient cohorts, and system health.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NoduleItem(BaseModel):
    """Detected nodule candidate candidate."""
    coord: List[float] = Field(..., description="[X, Y, Z] world millimeter coordinates")
    diameter_mm: float = Field(..., ge=0.0, description="Nodule diameter in millimeters")
    prob: float = Field(..., ge=0.0, le=1.0, description="Detection probability")


class PatientInfo(BaseModel):
    """Optional clinical and demographic patient details."""
    id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=130)
    sex: Optional[str] = None
    smoking_pack_years: Optional[int] = Field(None, ge=0)
    location: Optional[str] = None
    status: Optional[str] = None


class PredictRequest(BaseModel):
    """Request payload for CT volume nodule inference."""
    seriesuid: str = Field(..., description="Unique series identifier for CT scan")
    patient: Optional[Dict[str, Any]] = Field(default=None, description="Patient metadata")


class PredictResponse(BaseModel):
    """Standardized multi-task inference response."""
    seriesuid: str
    nodules: List[NoduleItem]
    top_nodule: Optional[NoduleItem] = None
    malignancy_score: int = Field(..., ge=0, le=5, description="Ordinal malignancy score (0=none, 1-5)")
    grade: str = Field(..., description="Roman numeral grade: 0, I, II, III, IV")
    grade_int: int = Field(..., ge=0, le=4, description="Integer grade 0-4")
    grade_label: str
    lung_rads: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    gradcam_slice_png_b64: str
    label_source: str = Field("proxy", description="'real' or 'proxy'")
    has_nodule: bool = Field(True, description="True when a focal nodule was identified")
    patient: Optional[Dict[str, Any]] = Field(default=None, description="Echoed patient metadata")
    disclaimer: str = Field("Research prototype. Not a medical device. Not for clinical diagnosis.")


class PatientRecord(BaseModel):
    """Cohort patient record schema."""
    model_config = {"protected_namespaces": ()}

    id: str
    name: str
    age: int
    sex: str
    smoking_pack_years: int
    scan_date: str
    seriesuid: str
    nodule_size_mm: float
    location: str
    malignancy_score: int
    model_grade: str
    model_grade_int: int
    grade_label: str
    lung_rads: str
    confidence: float
    recommendation: str
    status: str
    disclaimer: str


class StatsResponse(BaseModel):
    """Aggregate cohort and model validation statistics."""
    total_patients: int
    total_scans: int
    nodules_detected: int
    grade_distribution: Dict[str, int]
    mean_confidence: float
    high_risk_pct: float
    avg_nodules_per_scan: float
    val_sensitivity: Optional[float] = None
    val_specificity: Optional[float] = None
    label_source: str
    disclaimer: str = Field("Research prototype. Not a medical device. Not for clinical diagnosis.")


class HealthResponse(BaseModel):
    """Healthcheck response schema."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    label_source: str
    disclaimer: str = Field("Research prototype. Not a medical device. Not for clinical diagnosis.")
