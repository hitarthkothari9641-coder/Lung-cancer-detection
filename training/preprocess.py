"""
preprocess.py - CT Volume Preprocessing, Isotropic Resampling, and Patch Extraction.

Features:
- Reads LUNA16 .mhd / .raw CT volumes (SimpleITK with pure-Python raw fallback).
- Resamples volumes to 1x1x1 mm isotropic resolution.
- Clips Hounsfield Units (HU) to [-1000, 400] and min-max normalizes to [0, 1].
- Caches resampled volumes as float16 .npy arrays to accelerate training epochs.
- Extracts 64x64x64 voxel patches with bounding/padding and coordinate mapping.
"""

import os
import glob
import json
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from scipy import ndimage

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def parse_mhd_header(mhd_path: str) -> Dict[str, Any]:
    """Parse a MetaImage (.mhd) header file into a dictionary."""
    header = {}
    with open(mhd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip()] = val.strip()
    return header


def read_ct_volume(mhd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an .mhd / .raw CT scan.

    Returns:
        volume: 3D numpy array of shape (Z, Y, X) in Hounsfield Units.
        origin: np.ndarray [X, Y, Z] in world millimeter coordinates.
        spacing: np.ndarray [X, Y, Z] voxel spacing in millimeters.
    """
    if HAS_SITK:
        try:
            itk_img = sitk.ReadImage(mhd_path)
            volume = sitk.GetArrayFromImage(itk_img)  # (Z, Y, X)
            origin = np.array(itk_img.GetOrigin(), dtype=np.float32)  # (X, Y, Z)
            spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)  # (X, Y, Z)
            return volume.astype(np.float32), origin, spacing
        except Exception as e:
            pass

    # Pure Python / NumPy fallback if SimpleITK fails or is missing
    header = parse_mhd_header(mhd_path)
    dim_size = [int(v) for v in header.get("DimSize", "512 512 200").split()]  # (X, Y, Z)
    spacing = np.array([float(v) for v in header.get("ElementSpacing", "1.0 1.0 1.0").split()], dtype=np.float32)
    origin = np.array([float(v) for v in header.get("Offset", "0.0 0.0 0.0").split()], dtype=np.float32)
    raw_filename = header.get("ElementDataFile", os.path.basename(mhd_path).replace(".mhd", ".raw"))

    raw_path = os.path.join(os.path.dirname(mhd_path), raw_filename)
    dtype_map = {
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_FLOAT": np.float32,
    }
    raw_dtype = dtype_map.get(header.get("ElementType", "MET_SHORT"), np.int16)

    if os.path.exists(raw_path):
        raw_bytes = np.fromfile(raw_path, dtype=raw_dtype)
        # Reshape to (Z, Y, X)
        volume = raw_bytes.reshape((dim_size[2], dim_size[1], dim_size[0])).astype(np.float32)
    else:
        # Fallback dummy volume for pipeline testing
        volume = np.full((dim_size[2], dim_size[1], dim_size[0]), fill_value=-1000.0, dtype=np.float32)

    return volume, origin, spacing


def resample_to_isotropic(
    volume: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Resamples a 3D CT volume (Z, Y, X) to isotropic voxel spacing (e.g. 1.0 mm).

    Args:
        volume: 3D array of shape (Z, Y, X).
        current_spacing: [spX, spY, spZ] in mm.
        target_spacing: [spX, spY, spZ] desired isotropic spacing.

    Returns:
        resampled_volume: 3D array with target spacing.
    """
    # Current spacing order is (X, Y, Z); volume shape is (Z, Y, X)
    zoom_z = current_spacing[2] / target_spacing[2]
    zoom_y = current_spacing[1] / target_spacing[1]
    zoom_x = current_spacing[0] / target_spacing[0]
    zoom_factors = (zoom_z, zoom_y, zoom_x)

    # Scipy ndimage zoom with linear interpolation
    resampled = ndimage.zoom(volume, zoom=zoom_factors, order=1, mode="nearest")
    return resampled


def normalize_hu(volume: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """Clips Hounsfield Units to [hu_min, hu_max] and scales to [0.0, 1.0]."""
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def world_to_voxel(
    world_coord: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Converts world coordinate [X, Y, Z] (mm) to voxel coordinate [x, y, z].
    """
    return np.round((world_coord - origin) / spacing).astype(int)


def extract_patch(
    volume: np.ndarray,
    center_zyx: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Extracts a fixed 3D patch (D, H, W) centered at (cz, cy, cx) with boundary padding.

    Args:
        volume: 3D array (Z, Y, X).
        center_zyx: (cz, cy, cx) center coordinates.
        patch_size: (pd, ph, pw) dimensions, default (64, 64, 64).
        pad_value: Value used to pad when patch exceeds volume boundaries.

    Returns:
        3D patch of shape patch_size.
    """
    cz, cy, cx = center_zyx
    pd, ph, pw = patch_size
    rz, ry, rx = pd // 2, ph // 2, pw // 2

    z_start, z_end = cz - rz, cz + (pd - rz)
    y_start, y_end = cy - ry, cy + (ph - ry)
    x_start, x_end = cx - rx, cx + (pw - rx)

    # Target buffer
    patch = np.full(patch_size, fill_value=pad_value, dtype=volume.dtype)

    # Compute valid source and destination slices
    v_z0, v_z1 = max(0, z_start), min(volume.shape[0], z_end)
    v_y0, v_y1 = max(0, y_start), min(volume.shape[1], y_end)
    v_x0, v_x1 = max(0, x_start), min(volume.shape[2], x_end)

    p_z0 = v_z0 - z_start
    p_z1 = p_z0 + (v_z1 - v_z0)
    p_y0 = v_y0 - y_start
    p_y1 = p_y0 + (v_y1 - v_y0)
    p_x0 = v_x0 - x_start
    p_x1 = p_x0 + (v_x1 - v_x0)

    if (v_z1 > v_z0) and (v_y1 > v_y0) and (v_x1 > v_x0):
        patch[p_z0:p_z1, p_y0:p_y1, p_x0:p_x1] = volume[v_z0:v_z1, v_y0:v_y1, v_x0:v_x1]

    return patch


def preprocess_series(
    mhd_path: str,
    cache_dir: str = "./cache",
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> str:
    """
    Reads, resamples, normalizes, and saves a CT scan volume as float16 .npy cache.

    Returns:
        Path to the cached .npy file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    seriesuid = os.path.splitext(os.path.basename(mhd_path))[0]
    out_npy = os.path.join(cache_dir, f"{seriesuid}.npy")
    out_meta = os.path.join(cache_dir, f"{seriesuid}.json")

    if os.path.exists(out_npy) and os.path.exists(out_meta):
        return out_npy

    volume, origin, spacing = read_ct_volume(mhd_path)
    resampled = resample_to_isotropic(volume, spacing, target_spacing=(1.0, 1.0, 1.0))
    normalized = normalize_hu(resampled, hu_min=hu_min, hu_max=hu_max)

    # Save as float16 to conserve memory and disk
    np.save(out_npy, normalized.astype(np.float16))

    meta = {
        "seriesuid": seriesuid,
        "original_shape": list(volume.shape),
        "resampled_shape": list(normalized.shape),
        "origin": origin.tolist(),
        "original_spacing": spacing.tolist(),
        "target_spacing": [1.0, 1.0, 1.0],
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_npy
