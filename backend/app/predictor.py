"""
predictor.py - Production Predictor Module with TorchScript Loader and Mock Fallback.

Features:
- Checks for artifacts/model_ts.pt or artifacts/model.pt.
- If present, executes 3D TorchScript inference and Grad-CAM extraction.
- If absent, gracefully falls back to deterministic MockPredictor so all API endpoints
  and UI features function completely offline before training.
"""

import os
import sys
from typing import Dict, Any, Optional
import torch

from app.schemas import PredictResponse
from app.mock_predictor import MockPredictor
from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP


class PredictorService:
    """Manages model lifecycle and inference requests."""

    def __init__(
        self,
        artifacts_dir: str = "./artifacts",
        data_dir: str = "./data",
        device_str: str = "cpu",
    ):
        self.artifacts_dir = artifacts_dir
        self.data_dir = data_dir
        self.device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        self.model = None
        self.is_real_model = False
        self.label_source = "proxy"

        # Initialize fallback mock predictor
        sample_path = os.path.join(data_dir, "sample_patients.json")
        self.mock_predictor = MockPredictor(sample_patients_path=sample_path)

        self._check_and_load_model()

    def _check_and_load_model(self):
        """Attempts to load TorchScript or PyTorch checkpoint."""
        ts_path = os.path.join(self.artifacts_dir, "model_ts.pt")
        pt_path = os.path.join(self.artifacts_dir, "model.pt")
        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")

        if os.path.exists(metrics_path):
            try:
                import json
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    self.label_source = m.get("label_source", "proxy")
            except Exception:
                pass

        if os.path.exists(ts_path):
            try:
                self.model = torch.jit.load(ts_path, map_location=self.device)
                self.model.eval()
                self.is_real_model = True
                print(f"[PredictorService] Loaded TorchScript model from {ts_path}")
                return
            except Exception as e:
                print(f"[PredictorService] TorchScript load failed ({e}), checking PyTorch weights...")

        if os.path.exists(pt_path):
            try:
                from training.model import build_model
                net = build_model().to(self.device)
                state = torch.load(pt_path, map_location=self.device)
                net.load_state_dict(state)
                net.eval()
                self.model = net
                self.is_real_model = True
                print(f"[PredictorService] Loaded PyTorch checkpoint from {pt_path}")
                return
            except Exception as e:
                print(f"[PredictorService] PyTorch checkpoint load failed: {e}")

        print("[PredictorService] Model weights not found in artifacts/. Running in MOCK mode.")
        self.is_real_model = False

    def predict(
        self,
        seriesuid: str,
        patient_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on given seriesuid or patient metadata.
        Uses real predictor if loaded and scan exists, else deterministic mock predictor.
        """
        if self.is_real_model:
            # Check if .mhd exists
            mhd_candidates = [
                os.path.join(self.data_dir, "LUNA16", f"{seriesuid}.mhd"),
                os.path.join("./", f"{seriesuid}.mhd"),
            ]
            for i in range(10):
                mhd_candidates.append(os.path.join(self.data_dir, "LUNA16", f"subset{i}", f"{seriesuid}.mhd"))
                mhd_candidates.append(os.path.join("./", f"subset{i}", f"{seriesuid}.mhd"))

            mhd_path = next((p for p in mhd_candidates if os.path.exists(p)), None)
            if mhd_path is not None:
                from training.predict import predict_scan
                pred = predict_scan(mhd_path, patient_meta=patient_meta, device_str=str(self.device))
                pred["label_source"] = self.label_source
                return pred

        # Fallback to deterministic mock predictor
        return self.mock_predictor.predict(
            seriesuid=seriesuid,
            patient_meta=patient_meta,
            label_source=self.label_source,
        )
