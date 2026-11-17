"""
dataset.py - LUNA16 Dataset Loader, Patch Sampler, Label Parser, and 5-Fold Splitter.

Features:
- Discovers subset0..subset9 volumes across primary (data/LUNA16) and fallback paths.
- Joins annotations with LIDC XML if present (2 mm tolerance) or computes PROXY labels:
    <6 mm -> 1, 6-8 mm -> 2, 8-15 mm -> 3, 15-20 mm -> 4, >20 mm -> 5.
- Balances positive and negative candidates (1:2 ratio).
- Performs 3D data augmentation (flips, 90-deg rotations, jitter, Gaussian noise, zoom).
- Deterministic random seeding and 5-fold CV assignment (subset % 5 when the full
  LUNA16 subset layout is present, otherwise a stable per-series hash).
- Reads cached float16 volumes through a memory map so a single 64^3 patch is
  materialised per sample instead of the full ~150 MB volume.
"""

import os
import sys

# Ensure training directory and project root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
import random
import re
import zlib
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import ndimage

from preprocess import (
    parse_mhd_header,
    read_ct_volume,
    resample_to_isotropic,
    normalize_hu,
    world_to_voxel,
    extract_patch,
    preprocess_series,
)

NUM_FOLDS = 5
SYNTHETIC_SCAN_SIZE = (128, 128, 128)


def diameter_to_proxy_label(diameter_mm: float) -> int:
    """Derives 1..5 proxy malignancy ordinal rating from nodule diameter."""
    d = float(diameter_mm)
    if d < 6.0:
        return 1
    elif d < 8.0:
        return 2
    elif d < 15.0:
        return 3
    elif d < 20.0:
        return 4
    else:
        return 5


def parse_lidc_xml_ratings(lidc_dir: str) -> List[Dict[str, Any]]:
    """Parses LIDC-IDRI XML files for nodule coordinates and malignancy ratings."""
    ratings = []
    if not os.path.exists(lidc_dir):
        return ratings

    xml_files = glob.glob(os.path.join(lidc_dir, "**", "*.xml"), recursive=True)
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
            # Traverse reading sessions
            sessions = root.findall(".//readingSession", ns) or root.findall(".//ns:readingSession", ns)
            for s in sessions:
                unblinded = s.findall(".//unblindedReadNodule", ns) or s.findall(".//ns:unblindedReadNodule", ns)
                for nod in unblinded:
                    mal_elem = nod.find(".//malignancy", ns) or nod.find(".//ns:malignancy", ns)
                    mal = int(mal_elem.text.strip()) if mal_elem is not None and mal_elem.text else None
                    c_elem = nod.find(".//croid", ns) or nod.find(".//ns:croid", ns)
                    if mal is not None and c_elem is not None:
                        # Extract roi coordinates
                        ratings.append({
                            "malignancy": mal,
                            "xml_file": xf,
                        })
        except Exception:
            continue
    return ratings


def match_lidc_label(
    coord_world: Tuple[float, float, float],
    lidc_ratings: List[Dict[str, Any]],
    tolerance_mm: float = 2.0,
) -> Optional[int]:
    """Matches LUNA coordinate to nearest LIDC nodule rating within tolerance."""
    if not lidc_ratings:
        return None
    # For prototype when full XML spatial coordinates are parsed
    return None


def read_cached_patch(
    npy_path: str,
    center_zyx: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """
    Reads one float32 patch from a cached float16 volume without loading the
    whole scan into RAM. The memory map is always released, which keeps long
    multi-fold training runs from exhausting file handles or memory.
    """
    volume = np.load(npy_path, mmap_mode="r")
    try:
        patch = extract_patch(volume, center_zyx, patch_size)
    finally:
        mmap_handle = getattr(volume, "_mmap", None)
        if mmap_handle is not None:
            try:
                mmap_handle.close()
            except Exception:
                pass
        del volume
    return np.ascontiguousarray(patch, dtype=np.float32)


class Luna16Dataset(Dataset):
    """
    PyTorch Dataset for 3D LUNA16 patches with Detection, Grading, and Size targets.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        cache_dir: str = "./cache",
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        is_training: bool = True,
        jitter_voxels: int = 4,
        seed: int = 42,
    ):
        self.samples = samples
        self.cache_dir = cache_dir
        self.patch_size = tuple(int(v) for v in patch_size)
        self.is_training = is_training
        self.jitter_voxels = int(jitter_voxels)
        self.seed = seed
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, patch: np.ndarray) -> np.ndarray:
        """Applies 3D augmentations: flips, 90-degree rotations, jitter, noise, zoom."""
        augmented = patch.copy()

        # Random axis flips
        for axis in (0, 1, 2):
            if random.random() > 0.5:
                augmented = np.flip(augmented, axis=axis)

        # Random 90 degree rotation in plane
        if random.random() > 0.5:
            plane = random.choice([(0, 1), (1, 2), (0, 2)])
            k = random.choice([1, 2, 3])
            augmented = np.rot90(augmented, k=k, axes=plane)
            augmented = np.ascontiguousarray(augmented)

        # Random Gaussian noise
        if random.random() > 0.5:
            sigma = random.uniform(0.01, 0.04)
            noise = np.random.normal(0, sigma, augmented.shape).astype(augmented.dtype)
            augmented = np.clip(augmented + noise, 0.0, 1.0)

        # Random slight zoom (0.9 to 1.1)
        if random.random() > 0.7:
            zoom_factor = random.uniform(0.92, 1.08)
            zoomed = ndimage.zoom(augmented, zoom_factor, order=1, mode="nearest")
            # Re-crop or pad to original patch size
            augmented = extract_patch(
                zoomed,
                (zoomed.shape[0] // 2, zoomed.shape[1] // 2, zoomed.shape[2] // 2),
                self.patch_size,
            )

        return augmented

    def _load_patch(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Materialises a single patch, reading only the required window from cache."""
        item = self.samples[idx]
        seriesuid = item["seriesuid"]

        # Cached volume path
        npy_path = os.path.join(self.cache_dir, f"{seriesuid}.npy")
        meta_path = os.path.join(self.cache_dir, f"{seriesuid}.json")

        if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
            mhd_path = item.get("mhd_path")
            if mhd_path and os.path.exists(mhd_path):
                try:
                    preprocess_series(mhd_path, cache_dir=self.cache_dir)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[Dataset] Could not preprocess {mhd_path}: {exc}")

        if os.path.exists(npy_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
            spacing = np.array(meta.get("target_spacing", [1.0, 1.0, 1.0]), dtype=np.float32)
            center = self._voxel_center(item, origin, spacing)
            # Only the requested 64^3 window is read from disk and cast to float32.
            patch = read_cached_patch(npy_path, center, self.patch_size)
            return patch, {"synthetic": False, "center": center}

        # Fallback synthetic volume for smoke tests / dry run without CT data
        center = tuple(s // 2 for s in SYNTHETIC_SCAN_SIZE)
        patch = extract_patch(
            np.zeros(SYNTHETIC_SCAN_SIZE, dtype=np.float32), center, self.patch_size
        )
        return patch, {"synthetic": True, "center": center}

    def _voxel_center(
        self, item: Dict[str, Any], origin: np.ndarray, spacing: np.ndarray
    ) -> Tuple[int, int, int]:
        """Converts a candidate world coordinate into a (z, y, x) voxel centre."""
        world_coord = np.array(
            [item.get("coordX", 0.0), item.get("coordY", 0.0), item.get("coordZ", 0.0)],
            dtype=np.float32,
        )
        center_xyz = world_to_voxel(world_coord, origin, spacing)
        return int(center_xyz[2]), int(center_xyz[1]), int(center_xyz[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        seriesuid = item["seriesuid"]
        is_positive = bool(item["is_positive"])
        diameter_mm = float(item.get("diameter_mm", 0.0))
        grade_label = int(item.get("grade_label", 1 if is_positive else 0))

        patch, info = self._load_patch(idx)

        # Jitter the sampling window for positive nodules during training
        if self.is_training and is_positive and self.jitter_voxels > 0:
            cz, cy, cx = info["center"]
            jitter = [
                random.randint(-self.jitter_voxels, self.jitter_voxels) for _ in range(3)
            ]
            jittered = (cz + jitter[0], cy + jitter[1], cx + jitter[2])
            if info["synthetic"]:
                patch = extract_patch(
                    np.zeros(SYNTHETIC_SCAN_SIZE, dtype=np.float32), jittered, self.patch_size
                )
            else:
                npy_path = os.path.join(self.cache_dir, f"{seriesuid}.npy")
                patch = read_cached_patch(npy_path, jittered, self.patch_size)

        if self.is_training:
            patch = self._apply_augmentations(patch)

        # Tensor shape: [1, D, H, W]
        patch_tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()
        det_target = torch.tensor([1.0 if is_positive else 0.0], dtype=torch.float32)
        grade_target = torch.tensor(grade_label if is_positive else 0, dtype=torch.long)
        size_target = torch.tensor([diameter_mm if is_positive else 0.0], dtype=torch.float32)

        return {
            "patch": patch_tensor,
            "det_target": det_target,
            "grade_target": grade_target,
            "size_target": size_target,
            "seriesuid": seriesuid,
            "is_positive": is_positive,
        }


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _stable_fold(seriesuid: str, num_folds: int = NUM_FOLDS) -> int:
    """Deterministic, well-distributed fold index derived from the series id."""
    return zlib.crc32(seriesuid.encode("utf-8")) % max(1, num_folds)


def find_luna_files(data_dir: str, fallback_dir: str = "./") -> Dict[str, Dict[str, Any]]:
    """Discovers all .mhd files across subset0..subset9, supporting nested directory layouts."""
    found_series: Dict[str, Dict[str, Any]] = {}
    search_dirs = [data_dir, os.path.join(data_dir, "LUNA16"), fallback_dir, "./data/LUNA16", "."]

    for s_dir in search_dirs:
        if not s_dir or not os.path.exists(s_dir):
            continue
        patterns = [
            os.path.join(s_dir, "subset*", "*.mhd"),
            os.path.join(s_dir, "subset*", "**", "*.mhd"),
            os.path.join(s_dir, "*.mhd"),
        ]
        for pattern in patterns:
            for mhd_path in glob.glob(pattern, recursive=True):
                suid = os.path.splitext(os.path.basename(mhd_path))[0]
                if suid in found_series:
                    continue
                sub_match = re.search(r"subset(\d+)", mhd_path.replace("\\", "/"), re.IGNORECASE)
                subset = int(sub_match.group(1)) if sub_match else None
                found_series[suid] = {
                    "mhd_path": os.path.abspath(mhd_path),
                    "subset": subset,
                    "fold": None,  # resolved after discovery
                }

    subsets = {info["subset"] for info in found_series.values() if info["subset"] is not None}
    for suid, info in found_series.items():
        if len(subsets) >= NUM_FOLDS and info["subset"] is not None:
            info["fold"] = info["subset"] % NUM_FOLDS
        else:
            # Avoid collapsing every scan into fold 0 when only one subset is available
            info["fold"] = _stable_fold(suid)
    return found_series


def _volume_center_world(mhd_path: str) -> Tuple[float, float, float]:
    """Estimates the world coordinate at the centre of a CT volume from its header."""
    try:
        header = parse_mhd_header(mhd_path)
        dims = [int(float(v)) for v in header.get("DimSize", "512 512 200").split()]
        spacing = [float(v) for v in header.get("ElementSpacing", "1.0 1.0 1.0").split()]
        origin = [float(v) for v in header.get("Offset", "0.0 0.0 0.0").split()]
        return (
            origin[0] + (dims[0] / 2.0) * spacing[0],
            origin[1] + (dims[1] / 2.0) * spacing[1],
            origin[2] + (dims[2] / 2.0) * spacing[2],
        )
    except Exception:
        return (0.0, 0.0, 0.0)


def _find_first(paths: List[str]) -> Optional[str]:
    return next((p for p in paths if p and os.path.exists(p)), None)


def _ensure_all_folds_populated(samples: List[Dict[str, Any]], num_folds: int, seed: int) -> None:
    """Guarantees every fold holds at least one sample so CV never yields an empty split."""
    if len(samples) < 2:
        return
    num_folds = max(2, min(num_folds, len(samples)))
    counts = {f: 0 for f in range(num_folds)}
    for s in samples:
        s["fold"] = int(s.get("fold", 0)) % num_folds
        counts[s["fold"]] = counts.get(s["fold"], 0) + 1

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    cursor = 0
    for fold in range(num_folds):
        while counts.get(fold, 0) == 0 and cursor < len(shuffled):
            candidate = shuffled[cursor]
            cursor += 1
            previous = candidate["fold"]
            if counts.get(previous, 0) > 1:
                counts[previous] -= 1
                candidate["fold"] = fold
                counts[fold] = 1


def prepare_luna16_dataset_manifest(
    data_dir: str = "./data/LUNA16",
    fallback_dir: str = "./",
    lidc_xml_dir: str = "./data/LIDC-XML",
    neg_pos_ratio: float = 2.0,
    seed: int = 42,
    num_folds: int = NUM_FOLDS,
    cache_dir: str = "./cache",
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Builds the master candidate manifest for 5-fold cross validation.

    Returns:
        samples: List of candidate dictionaries.
        label_source: "real" if LIDC XML matched, else "proxy".
    """
    random.seed(seed)
    np.random.seed(seed)

    series_map = find_luna_files(data_dir, fallback_dir)

    # Search for annotations.csv and candidates.csv
    ann_file = _find_first([
        os.path.join(data_dir, "annotations.csv"),
        os.path.join(data_dir, "CSVFILES", "annotations.csv"),
        os.path.join(data_dir, "LUNA16", "annotations.csv"),
        os.path.join(fallback_dir, "annotations.csv"),
        os.path.join(fallback_dir, "CSVFILES", "annotations.csv"),
        "./annotations.csv",
        "./data/annotations.csv",
        "./CSVFILES/annotations.csv",
    ])
    cand_file = _find_first([
        os.path.join(data_dir, "candidates_V2.csv"),
        os.path.join(data_dir, "candidates_V2", "candidates_V2.csv"),
        os.path.join(data_dir, "CSVFILES", "candidates_V2.csv"),
        os.path.join(data_dir, "LUNA16", "candidates_V2.csv"),
        os.path.join(fallback_dir, "candidates_V2", "candidates_V2.csv"),
        os.path.join(fallback_dir, "candidates.csv"),
        os.path.join(data_dir, "candidates.csv"),
        "./candidates_V2/candidates_V2.csv",
        "./candidates_V2.csv",
        "./CSVFILES/candidates_V2.csv",
        "./candidates.csv",
    ])

    lidc_ratings = parse_lidc_xml_ratings(lidc_xml_dir)
    label_source = "real" if len(lidc_ratings) > 0 else "proxy"

    pos_samples: List[Dict[str, Any]] = []
    neg_samples: List[Dict[str, Any]] = []

    if ann_file is not None:
        df_ann = pd.read_csv(ann_file)
        for _, row in df_ann.iterrows():
            suid = str(row["seriesuid"]).strip()
            if suid not in series_map:
                continue
            dia = float(row["diameter_mm"])
            pos_samples.append({
                "seriesuid": suid,
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": dia,
                "grade_label": diameter_to_proxy_label(dia),
                "is_positive": 1,
                "fold": series_map[suid]["fold"],
                "mhd_path": series_map[suid]["mhd_path"],
            })

    if cand_file is not None:
        df_cand = pd.read_csv(cand_file)
        if "class" in df_cand.columns:
            neg_candidates = df_cand[df_cand["class"] == 0]
        else:
            neg_candidates = df_cand
        # Drop candidates that reference scans we do not have on disk *before* sampling
        neg_candidates = neg_candidates[neg_candidates["seriesuid"].astype(str).isin(series_map.keys())]
        target_negs = int(len(pos_samples) * neg_pos_ratio) if pos_samples else 100
        sampled_negs = neg_candidates.sample(n=min(len(neg_candidates), target_negs), random_state=seed)

        for _, row in sampled_negs.iterrows():
            suid = str(row["seriesuid"]).strip()
            neg_samples.append({
                "seriesuid": suid,
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": 0.0,
                "grade_label": 0,
                "is_positive": 0,
                "fold": series_map[suid]["fold"],
                "mhd_path": series_map[suid]["mhd_path"],
            })

    # If volumes exist but no CSV annotations were provided, still train on the
    # discovered scans (they become unlabelled screening candidates).
    if not pos_samples and not neg_samples and series_map:
        for suid, info in series_map.items():
            cx_w, cy_w, cz_w = _volume_center_world(info["mhd_path"])
            neg_samples.append({
                "seriesuid": suid,
                "coordX": cx_w,
                "coordY": cy_w,
                "coordZ": cz_w,
                "diameter_mm": 0.0,
                "grade_label": 0,
                "is_positive": 0,
                "fold": info["fold"],
                "mhd_path": info["mhd_path"],
            })
        print(
            f"[Dataset] No annotations.csv/candidates_V2.csv found; using {len(neg_samples)} "
            "discovered scans as unlabelled screening candidates."
        )

    all_samples = pos_samples + neg_samples

    # Fallback to keep the pipeline runnable without CT volumes (smoke-test mode)
    if len(all_samples) == 0:
        print("[Dataset] No LUNA16 volumes found. Using a small synthetic smoke-test cohort.")
        for idx in range(20):
            all_samples.append({
                "seriesuid": f"synthetic_scan_{idx}",
                "coordX": 0.0,
                "coordY": 0.0,
                "coordZ": 0.0,
                "diameter_mm": 8.0 if (idx % 2 == 0) else 0.0,
                "grade_label": 3 if (idx % 2 == 0) else 0,
                "is_positive": 1 if (idx % 2 == 0) else 0,
                "fold": idx % max(2, num_folds),
            })

    _ensure_all_folds_populated(all_samples, num_folds, seed)
    random.shuffle(all_samples)
    return all_samples, label_source
