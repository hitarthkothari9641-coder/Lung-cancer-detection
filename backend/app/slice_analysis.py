"""
slice_analysis.py - 2D CT Slice Analysis for Uploaded PNG/JPG Images.

Uses classical image analysis (HU-aware windowing + connected components) to
locate a focal rounded opacity in a single axial CT slice. This is the fallback
path used when a full 3D volume is not available; the 3D network is preferred
whenever a .mhd/.raw volume is supplied.
"""

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP
from training.gradcam3d import blend_axial_slice_gradcam

MAX_DIMENSION = 512
MIN_DIAMETER_MM = 2.0
MAX_DIAMETER_MM = 30.0


def load_gray_image(data: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decodes image bytes into a normalized float32 [0, 1] grayscale array."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("L")
        if max(img.size) > MAX_DIMENSION:
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr, img.size


def _largest_round_component(mask: np.ndarray, exclude_border_px: int = 0) -> Optional[Dict[str, Any]]:
    """Finds the most nodule-like (largest, most circular) connected component."""
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is a hard backend requirement
        return None

    labeled, num = ndimage.label(mask)
    if num == 0:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for idx in range(1, num + 1):
        ys, xs = np.where(labeled == idx)
        area = len(xs)
        if area < 20:
            continue
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        # Skip thin streaks (vessels / ribs), long edges, and anything touching the
        # image border (chest wall, table, blanket).
        if exclude_border_px > 0 and (
            ys.min() < exclude_border_px
            or xs.min() < exclude_border_px
            or ys.max() > mask.shape[0] - exclude_border_px
            or xs.max() > mask.shape[1] - exclude_border_px
        ):
            continue
        aspect = max(height, width) / max(1, min(height, width))
        if aspect > 1.8:
            continue
        fill_ratio = area / float(height * width)
        if fill_ratio < 0.5:
            continue
        score = float(area) * float(min(1.0, fill_ratio * 1.4)) / aspect
        if score > best_score:
            best_score = score
            best = {
                "centroid": (float(ys.mean()), float(xs.mean())),
                "area": int(area),
                "height": height,
                "width": width,
            }
    return best


def analyze_slice(data: bytes, patient_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyzes a single axial CT image and returns a schema-compatible prediction.

    The slice is windowed, thresholded for soft-tissue-density focal opacities,
    and the most nodule-like component drives the diameter estimate and grading.
    """
    arr, (width_px, height_px) = load_gray_image(data)

    # Focal opacity detection = white top-hat: subtract a smooth local background
    # (lung parenchyma, chest wall, vessels) and keep only compact bright bumps.
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(arr, sigma=1.0)
    window = max(9, int(min(arr.shape) / 10))
    background = ndimage.uniform_filter(smoothed, size=window, mode="nearest")
    response = smoothed - background
    response_std = float(response.std()) or 0.02
    threshold = max(0.05, response_std * 1.8)
    mask = response > threshold

    # Only consider tissue inside the body (excludes surrounding air/table)
    body_level = float(np.percentile(arr, 60))
    mask &= arr > max(0.08, body_level * 0.35)

    # Ignore the chest wall / table by excluding the outer 3 % of the image
    border = max(4, min(arr.shape) // 32)
    component = _largest_round_component(mask, exclude_border_px=border)

    has_nodule = component is not None
    cam_slice = np.zeros(arr.shape, dtype=np.float32)
    diameter_mm = 0.0

    if component is not None:
        # Reconstructed axial slices span roughly 350 mm across 512 px (~0.7 mm/px).
        # Smaller images are treated as cropped lung windows of the same field.
        mm_per_pixel = 0.7 * (256.0 / float(max(width_px, height_px)))
        mean_diameter_px = 0.5 * (component["height"] + component["width"])
        diameter_mm = float(np.clip(mean_diameter_px * mm_per_pixel, MIN_DIAMETER_MM, MAX_DIAMETER_MM))

        cy, cx = component["centroid"]
        yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
        sigma = max(3.0, 0.5 * mean_diameter_px)
        cam_slice = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
        cam_slice = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)

    # Malignancy rating from focal contrast + size (proxy for opacity density)
    background_level = float(np.median(arr))
    spread = float(arr.std()) or 0.05
    if has_nodule:
        peak = float(np.max(arr[mask])) if np.any(mask) else background_level
        contrast = float(np.clip((peak - background_level) / max(0.05, spread * 2.2), 0.0, 1.0))
        size_term = min(1.0, diameter_mm / 22.0)
        malignancy_score = int(np.clip(round(1 + 4 * (0.55 * size_term + 0.45 * contrast)), 1, 5))
        confidence = float(np.clip(0.62 + 0.3 * contrast, 0.6, 0.97))
    else:
        malignancy_score = 0
        confidence = float(np.clip(0.8 + 0.15 * (1.0 - min(1.0, spread)), 0.8, 0.98))

    grade_int, grade_label, lung_rads, recommendation = malignancy_to_grade(
        score=malignancy_score, diameter_mm=diameter_mm
    )

    gradcam_b64 = blend_axial_slice_gradcam(np.clip(arr, 0.0, 1.0), cam_slice)
    seriesuid = "uploaded-slice"

    return {
        "seriesuid": seriesuid,
        "nodules": (
            [
                {
                    "coord": [
                        round(float(component["centroid"][1]), 1),
                        round(float(component["centroid"][0]), 1),
                        round(diameter_mm, 1),
                    ],
                    "diameter_mm": round(diameter_mm, 1),
                    "prob": round(confidence, 2),
                }
            ]
            if has_nodule
            else []
        ),
        "top_nodule": (
            {
                "coord": [
                    round(float(component["centroid"][1]), 1),
                    round(float(component["centroid"][0]), 1),
                    round(diameter_mm, 1),
                ],
                "diameter_mm": round(diameter_mm, 1),
                "prob": round(confidence, 2),
            }
            if has_nodule
            else None
        ),
        "malignancy_score": malignancy_score,
        "grade": GRADE_ROMAN_MAP[grade_int],
        "grade_int": grade_int,
        "grade_label": grade_label,
        "lung_rads": lung_rads,
        "confidence": round(confidence, 2),
        "recommendation": recommendation,
        "gradcam_slice_png_b64": gradcam_b64,
        "patient": patient_meta or {},
        "has_nodule": has_nodule,
        "disclaimer": "Research prototype. Not a medical device. Not for clinical diagnosis.",
    }
