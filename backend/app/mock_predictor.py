"""
mock_predictor.py - Deterministic Mock Inference Engine.

Used for offline demonstration, UI development, and testing when model weights are not loaded.
Strictly preserves consistency with sample_patients.json and backend/app/grading.py.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP
from training.gradcam3d import generate_mock_ct_gradcam_slice


class MockPredictor:
    """Deterministic mock predictor for offline execution."""

    def __init__(self, sample_patients_path: str = "./data/sample_patients.json"):
        self.sample_patients_path = sample_patients_path
        self.patient_lookup: Dict[str, Dict[str, Any]] = {}
        self._load_sample_patients()

    def _load_sample_patients(self):
        # Look in relative paths
        paths_to_check = [
            self.sample_patients_path,
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_patients.json"),
            "./data/sample_patients.json",
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    for r in records:
                        self.patient_lookup[r["seriesuid"]] = r
                        self.patient_lookup[r["id"]] = r
                    break
                except Exception:
                    pass

    def predict(
        self,
        seriesuid: str,
        patient_meta: Optional[Dict[str, Any]] = None,
        label_source: str = "proxy",
    ) -> Dict[str, Any]:
        """Generates deterministic prediction results matching grading.py."""
        # Check if known seriesuid or patient
        match = self.patient_lookup.get(seriesuid)
        if match is None and patient_meta and "id" in patient_meta:
            match = self.patient_lookup.get(patient_meta["id"])

        if match is not None:
            score = match.get("malignancy_score", 3)
            size_mm = float(match.get("nodule_size_mm", 8.0))
            confidence = float(match.get("confidence", 0.85))
            has_nodule = size_mm > 0.0
        else:
            # Deterministic hash of unknown seriesuid
            h = int(hashlib.sha256(seriesuid.encode("utf-8")).hexdigest()[:8], 16)
            score = (h % 5) + 1
            has_nodule = (h % 6) != 0  # ~16% chance of no nodule
            if not has_nodule:
                score = 0
                size_mm = 0.0
                confidence = 0.95
            else:
                size_buckets = [4.5, 7.2, 11.5, 17.8, 24.0]
                size_mm = size_buckets[(h // 5) % len(size_buckets)]
                confidence = 0.65 + ((h % 30) / 100.0)

        # Single source of truth: clinical rule overlay
        grade_int, grade_label, lung_rads, recommendation = malignancy_to_grade(score, size_mm)
        grade_roman = GRADE_ROMAN_MAP[grade_int]

        # Grad-CAM slice
        gradcam_b64 = generate_mock_ct_gradcam_slice(
            diameter_mm=size_mm,
            has_nodule=has_nodule,
        )

        nodules = []
        top_nodule = None

        if has_nodule:
            # Deterministic coordinates
            coord = [
                round(35.2 + (score * 2.1), 2),
                round(-42.8 - (score * 1.5), 2),
                round(-180.5 + (score * 4.2), 2),
            ]
            top_nodule = {
                "coord": coord,
                "diameter_mm": size_mm,
                "prob": round(confidence, 2),
            }
            nodules.append(top_nodule)

        return {
            "seriesuid": seriesuid,
            "nodules": nodules,
            "top_nodule": top_nodule,
            "malignancy_score": score,
            "grade": grade_roman,
            "grade_int": grade_int,
            "grade_label": grade_label,
            "lung_rads": lung_rads,
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "gradcam_slice_png_b64": gradcam_b64,
            "label_source": label_source,
            "disclaimer": "Research prototype. Not a medical device. Not for clinical diagnosis.",
        }
