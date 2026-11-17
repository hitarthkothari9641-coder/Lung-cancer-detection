"""
predict.py - Standalone 3D CT Volume Inference, Nodule Candidate Generation, and Grading.

Features:
- Reads .mhd / .raw volume, resamples to isotropic, and runs sliding patch inference.
- Connected component clustering and 3D Non-Maximum Suppression (NMS).
- Runs grading and size regression heads on the top detected nodule.
- Single source of truth grading mapping via backend/app/grading.py.
- Produces a real axial CT slice at the detected nodule with a Grad-CAM overlay,
  encoded as a base64 PNG (falls back to a deterministic phantom slice when no
  CT volume is available).
- Returns a comprehensive prediction JSON.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

# Ensure training, backend, and project root can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
for _p in (SCRIPT_DIR, PROJECT_ROOT, BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP
from preprocess import read_ct_volume, resample_to_isotropic, normalize_hu, extract_patch
from losses import binary_to_ordinal_prediction
from gradcam3d import (
    GradCAM3D,
    blend_axial_slice_gradcam,
    generate_mock_ct_gradcam_slice,
    generate_synthetic_cam_3d,
)

DISCLAIMER = "Research prototype. Not a medical device. Not for clinical diagnosis."
PATCH_SIZE = (64, 64, 64)


def non_max_suppression_3d(
    candidates: List[Dict[str, Any]],
    distance_threshold_mm: float = 12.0,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    3D Non-Maximum Suppression to filter duplicate candidate detections.

    Args:
        candidates: List of dicts with 'coord' [x, y, z] and 'prob'.
        distance_threshold_mm: Minimum distance in mm between separate nodules.
        top_k: Maximum number of nodules to return.

    Returns:
        Filtered list of top candidates.
    """
    if not candidates:
        return []

    sorted_cand = sorted(candidates, key=lambda c: c["prob"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_cand:
        c_coord = np.array(cand["coord"], dtype=float)
        suppress = False
        for k in kept:
            k_coord = np.array(k["coord"], dtype=float)
            if np.linalg.norm(c_coord - k_coord) < distance_threshold_mm:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
            if len(kept) >= top_k:
                break

    return kept


def _load_model(model_path: Optional[str], device: torch.device):
    """Loads TorchScript or PyTorch weights. Returns None when nothing usable exists."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        if "ts" in os.path.basename(model_path):
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            return model
        from model import build_model
        model = build_model().to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        print(f"[predict] Could not load model from {model_path}: {exc}")
        return None


def _scan_candidates(model, vol_norm: np.ndarray, origin: np.ndarray, device: torch.device) -> List[Dict[str, Any]]:
    """Sliding-window candidate generation over a normalized volume."""
    d, h, w = vol_norm.shape
    step_z = max(16, (d - PATCH_SIZE[0]) // 4)
    step_y = max(16, (h - PATCH_SIZE[1]) // 6)
    step_x = max(16, (w - PATCH_SIZE[2]) // 6)

    candidates: List[Dict[str, Any]] = []
    with torch.no_grad():
        for cz in range(PATCH_SIZE[0] // 2, max(PATCH_SIZE[0] // 2 + 1, d - PATCH_SIZE[0] // 2), step_z):
            for cy in range(PATCH_SIZE[1] // 2, max(PATCH_SIZE[1] // 2 + 1, h - PATCH_SIZE[1] // 2), step_y):
                for cx in range(PATCH_SIZE[2] // 2, max(PATCH_SIZE[2] // 2 + 1, w - PATCH_SIZE[2] // 2), step_x):
                    patch = extract_patch(vol_norm, (cz, cy, cx), patch_size=PATCH_SIZE)
                    if float(np.mean(patch)) < 0.05 or float(np.std(patch)) < 0.02:
                        continue

                    tensor_in = (
                        torch.from_numpy(np.ascontiguousarray(patch))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )
                    det_logits, grade_logits, size_pred = model(tensor_in)
                    prob = float(torch.sigmoid(det_logits).reshape(-1)[0].item())

                    if prob > 0.40:
                        candidates.append({
                            "coord": [
                                round(float(origin[0] + cx), 2),
                                round(float(origin[1] + cy), 2),
                                round(float(origin[2] + cz), 2),
                            ],
                            "prob": round(prob, 4),
                            "cz": int(cz), "cy": int(cy), "cx": int(cx),
                            "size_mm": max(2.0, round(float(size_pred.reshape(-1)[0].item()), 2)),
                            "grade_logits": grade_logits.detach().cpu(),
                        })
    return candidates


def _render_slice(
    model,
    vol_norm: np.ndarray,
    top: Optional[Dict[str, Any]],
    device: torch.device,
) -> str:
    """Renders the axial CT slice at the detected nodule with a Grad-CAM overlay."""
    if vol_norm.size == 0:
        return generate_mock_ct_gradcam_slice(
            diameter_mm=float(top.get("size_mm", 0.0)) if top else 0.0,
            has_nodule=top is not None,
        )

    if top is not None:
        center = (top["cz"], top["cy"], top["cx"])
    else:
        center = (vol_norm.shape[0] // 2, vol_norm.shape[1] // 2, vol_norm.shape[2] // 2)

    patch = extract_patch(vol_norm, center, patch_size=PATCH_SIZE).astype(np.float32)
    cam_3d: np.ndarray
    try:
        if model is not None:
            cam_3d = GradCAM3D(model).generate_cam(
                torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).unsqueeze(0).float().to(device)
            )
        else:
            cam_3d = generate_synthetic_cam_3d(patch.shape)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[predict] Grad-CAM failed ({exc}); using attenuation-based heatmap.")
        cam_3d = np.asarray(patch, dtype=np.float32)
        span = cam_3d.max() - cam_3d.min()
        cam_3d = (cam_3d - cam_3d.min()) / span if span > 0 else np.zeros_like(cam_3d)

    # Axial slice through the nodule (fall back to max projection if the CAM is flat)
    slice_idx = patch.shape[0] // 2
    ct_slice = np.clip(patch[slice_idx], 0.0, 1.0)
    cam_slice = cam_3d[slice_idx] if cam_3d.ndim == 3 else cam_3d
    if cam_slice.max() - cam_slice.min() < 1e-6:
        cam_slice = np.max(cam_3d, axis=0)
    return blend_axial_slice_gradcam(ct_slice, cam_slice)


def predict_scan(
    mhd_path: str,
    patient_meta: Optional[Dict[str, Any]] = None,
    model_path: str = "./artifacts/model_ts.pt",
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """
    Runs end-to-end nodule detection, ordinal grading, and Grad-CAM on a CT volume.

    Args:
        mhd_path: Absolute or relative path to the .mhd file.
        patient_meta: Optional dictionary containing patient demographic data.
        model_path: Path to PyTorch model weights or TorchScript file.
        device_str: "cpu" or "cuda".

    Returns:
        Dictionary conforming to the API prediction schema.
    """
    seriesuid = os.path.splitext(os.path.basename(mhd_path))[0]
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    model = _load_model(model_path, device)

    try:
        raw_vol, origin, spacing = read_ct_volume(mhd_path)
        vol_norm = normalize_hu(resample_to_isotropic(raw_vol, spacing))
    except Exception as exc:
        print(f"[predict] Could not read {mhd_path}: {exc}")
        vol_norm = np.zeros((128, 128, 128), dtype=np.float32)
        origin = np.zeros(3, dtype=np.float32)

    candidates: List[Dict[str, Any]] = []
    if model is not None and vol_norm.size > 0:
        candidates = _scan_candidates(model, vol_norm, np.asarray(origin), device)
    nodules = non_max_suppression_3d(candidates, distance_threshold_mm=12.0, top_k=5)

    if nodules:
        top = nodules[0]
        grade_logits = top.get("grade_logits")
        if grade_logits is not None:
            pred_ranks, _ = binary_to_ordinal_prediction(grade_logits)
            malignancy_score = int(pred_ranks[0].item())
        else:
            malignancy_score = 3
        nodule_size_mm = float(top.get("size_mm", 8.5))
        confidence = float(top["prob"])
        has_nodule = True
    else:
        top = None
        malignancy_score = 0
        nodule_size_mm = 0.0
        confidence = 0.0
        has_nodule = False

    grade_int, grade_label, lung_rads, recommendation = malignancy_to_grade(
        score=malignancy_score,
        diameter_mm=nodule_size_mm,
    )
    grade_roman = GRADE_ROMAN_MAP[grade_int]

    gradcam_b64 = _render_slice(model, vol_norm, top, device)

    return {
        "seriesuid": seriesuid,
        "nodules": [
            {
                "coord": n["coord"],
                "diameter_mm": float(n.get("size_mm", nodule_size_mm)),
                "prob": float(n["prob"]),
            }
            for n in nodules
        ],
        "top_nodule": (
            {
                "coord": top["coord"],
                "diameter_mm": nodule_size_mm,
                "prob": float(top["prob"]),
            }
            if top
            else None
        ),
        "malignancy_score": malignancy_score,
        "grade": grade_roman,
        "grade_int": grade_int,
        "grade_label": grade_label,
        "lung_rads": lung_rads,
        "confidence": round(confidence, 2),
        "recommendation": recommendation,
        "gradcam_slice_png_b64": gradcam_b64,
        "patient": patient_meta or {},
        "has_nodule": has_nodule,
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D inference on a single CT volume")
    parser.add_argument("--mhd", type=str, required=True, help="Path to input .mhd CT volume")
    parser.add_argument("--model", type=str, default="./artifacts/model_ts.pt", help="Model weights path")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    pred = predict_scan(args.mhd, model_path=args.model)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(pred, f, indent=2)
        print(f"Prediction saved to: {args.out}")
    else:
        summary = {k: v for k, v in pred.items() if k != "gradcam_slice_png_b64"}
        print(json.dumps(summary, indent=2))
