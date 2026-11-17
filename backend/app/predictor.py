"""
predictor.py - Production Predictor Module with TorchScript Loader and Mock Fallback.

Features:
- Loads artifacts/model_ts.pt (preferred) or artifacts/model.pt once at startup.
- Runs the real 3D pipeline for scanned volumes, including uploaded .mhd/.raw files.
- Falls back to the deterministic MockPredictor so every endpoint works before training.
"""

import os
import sys
from typing import Dict, Any, Optional
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.mock_predictor import MockPredictor


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
        self.upload_dir = os.path.join(artifacts_dir, "uploads")
        self.device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        self.model = None
        self.is_real_model = False
        self.label_source = "proxy"

        sample_path = os.path.join(data_dir, "sample_patients.json")
        self.mock_predictor = MockPredictor(sample_patients_path=sample_path)

        self._check_and_load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _check_and_load_model(self) -> None:
        """Attempts to load TorchScript or PyTorch checkpoint exactly once."""
        ts_path = os.path.join(self.artifacts_dir, "model_ts.pt")
        pt_path = os.path.join(self.artifacts_dir, "model.pt")
        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")

        if os.path.exists(metrics_path):
            try:
                import json
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self.label_source = json.load(f).get("label_source", "proxy")
            except Exception:
                pass

        if os.path.exists(ts_path):
            try:
                model = torch.jit.load(ts_path, map_location=self.device)
                model.eval()
                self.model = model
                self.is_real_model = True
                print(f"[PredictorService] Loaded TorchScript model from {ts_path}")
                return
            except Exception as exc:
                print(f"[PredictorService] TorchScript load failed ({exc}), checking checkpoints...")

        if os.path.exists(pt_path):
            try:
                from training.model import build_model
                net = build_model().to(self.device)
                state = torch.load(pt_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                net.load_state_dict(state)
                net.eval()
                self.model = net
                self.is_real_model = True
                print(f"[PredictorService] Loaded PyTorch checkpoint from {pt_path}")
                return
            except Exception as exc:
                print(f"[PredictorService] PyTorch checkpoint load failed: {exc}")

        print("[PredictorService] Model weights not found in artifacts/. Running in demo mode.")
        self.is_real_model = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _resolve_volume_path(self, seriesuid: str) -> Optional[str]:
        """Finds a CT volume for a seriesuid, including previously uploaded scans."""
        candidates = [
            os.path.join(self.upload_dir, f"{seriesuid}.mhd"),
            os.path.join(self.upload_dir, f"{seriesuid}.raw"),
            os.path.join(self.upload_dir, f"{seriesuid}.dcm"),
            os.path.join(self.upload_dir, f"{seriesuid}.nii"),
            os.path.join(self.data_dir, "LUNA16", f"{seriesuid}.mhd"),
            os.path.join(PROJECT_ROOT, f"{seriesuid}.mhd"),
            os.path.join(self.data_dir, f"{seriesuid}.mhd"),
        ]
        for i in range(10):
            candidates.append(os.path.join(self.data_dir, "LUNA16", f"subset{i}", f"{seriesuid}.mhd"))
            candidates.append(os.path.join(PROJECT_ROOT, f"subset{i}", f"{seriesuid}.mhd"))
        return next((p for p in candidates if os.path.exists(p)), None)

    def predict(
        self,
        seriesuid: str,
        patient_meta: Optional[Dict[str, Any]] = None,
        mhd_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on a seriesuid or an explicit volume path.

        Uses the trained 3D model when both weights and a volume are available;
        otherwise returns the deterministic demo prediction.
        """
        if self.is_real_model:
            volume_path = mhd_path if mhd_path and os.path.exists(mhd_path) else self._resolve_volume_path(seriesuid)
            if volume_path is not None:
                try:
                    from training.predict import predict_scan
                    # Prefer the portable TorchScript export, fall back to the checkpoint
                    weights = os.path.join(self.artifacts_dir, "model_ts.pt")
                    if not os.path.exists(weights):
                        weights = os.path.join(self.artifacts_dir, "model.pt")
                    prediction = predict_scan(
                        volume_path,
                        patient_meta=patient_meta,
                        model_path=weights,
                        device_str=str(self.device),
                    )
                    prediction["seriesuid"] = seriesuid
                    prediction["label_source"] = self.label_source
                    return prediction
                except Exception as exc:
                    print(f"[PredictorService] 3D inference failed ({exc}); using demo prediction.")

        prediction = self.mock_predictor.predict(
            seriesuid=seriesuid,
            patient_meta=patient_meta,
            label_source=self.label_source,
        )
        return prediction
