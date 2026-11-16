"""
test_schemas.py - Unit Tests for Pydantic API Schemas and Payload Validation.
"""

import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import (
    NoduleItem,
    PredictRequest,
    PredictResponse,
    PatientRecord,
    StatsResponse,
    HealthResponse,
)


def test_nodule_item_valid():
    """Verify valid nodule schema construction."""
    nodule = NoduleItem(coord=[12.5, -45.0, 110.2], diameter_mm=8.4, prob=0.88)
    assert nodule.diameter_mm == 8.4
    assert nodule.prob == 0.88
    assert len(nodule.coord) == 3


def test_nodule_item_invalid_prob():
    """Verify validation fails when probability is outside [0, 1]."""
    with pytest.raises(ValidationError):
        NoduleItem(coord=[0.0, 0.0, 0.0], diameter_mm=5.0, prob=1.5)

    with pytest.raises(ValidationError):
        NoduleItem(coord=[0.0, 0.0, 0.0], diameter_mm=-2.0, prob=0.5)


def test_predict_request_validation():
    """Verify predict request schema parsing."""
    req = PredictRequest(seriesuid="1.3.6.1.4.1.14519.5.2.1.6279.6001.1001")
    assert req.seriesuid == "1.3.6.1.4.1.14519.5.2.1.6279.6001.1001"
    assert req.patient is None


def test_predict_response_validation():
    """Verify full prediction response contract."""
    resp = PredictResponse(
        seriesuid="test.series.123",
        nodules=[NoduleItem(coord=[1.0, 2.0, 3.0], diameter_mm=7.0, prob=0.9)],
        top_nodule=NoduleItem(coord=[1.0, 2.0, 3.0], diameter_mm=7.0, prob=0.9),
        malignancy_score=3,
        grade="II",
        grade_int=2,
        grade_label="Probably benign",
        lung_rads="Lung-RADS 3",
        confidence=0.90,
        recommendation="6-month follow-up CT",
        gradcam_slice_png_b64="data:image/png;base64,mockdata",
        label_source="proxy",
    )
    assert resp.grade == "II"
    assert resp.grade_int == 2
    assert "Not a medical device" in resp.disclaimer


def test_stats_response_validation():
    """Verify stats response structure."""
    stats = StatsResponse(
        total_patients=12,
        total_scans=12,
        nodules_detected=10,
        grade_distribution={"0": 2, "I": 3, "II": 2, "III": 3, "IV": 2},
        mean_confidence=0.84,
        high_risk_pct=41.7,
        avg_nodules_per_scan=0.83,
        val_sensitivity=0.862,
        val_specificity=0.915,
        label_source="proxy",
    )
    assert stats.total_patients == 12
    assert stats.grade_distribution["IV"] == 2
