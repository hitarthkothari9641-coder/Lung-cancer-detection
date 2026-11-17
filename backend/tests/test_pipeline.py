"""
test_pipeline.py - Regression tests for the preprocessing -> training -> serving path.

These cover the defects that previously broke Step 2 of training and the upload
endpoint, so they cannot silently come back.
"""

import io
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
for _p in (PROJECT_ROOT, TRAINING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import (  # noqa: E402
    Luna16Dataset,
    prepare_luna16_dataset_manifest,
    read_cached_patch,
    diameter_to_proxy_label,
)
from preprocess import extract_patch  # noqa: E402
import gradcam3d  # noqa: E402


@pytest.fixture()
def cached_scan(tmp_path):
    """Writes a small float16 cached volume plus its metadata sidecar."""
    volume = np.linspace(0, 1, 32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32)
    npy_path = tmp_path / "scan1.npy"
    np.save(npy_path, volume.astype(np.float16))
    (tmp_path / "scan1.json").write_text(
        json.dumps({"origin": [0, 0, 0], "target_spacing": [1, 1, 1]}),
        encoding="utf-8",
    )
    return str(tmp_path)


def _sample(**overrides):
    base = {
        "seriesuid": "scan1",
        "coordX": 16.0,
        "coordY": 16.0,
        "coordZ": 16.0,
        "diameter_mm": 8.0,
        "grade_label": 3,
        "is_positive": 1,
        "fold": 0,
    }
    base.update(overrides)
    return base


def test_dataset_reads_json_metadata_without_name_error(cached_scan):
    """Regression: missing `import json` raised NameError and broke Step 2."""
    dataset = Luna16Dataset([_sample()], cache_dir=cached_scan, is_training=False)
    item = dataset[0]

    assert item["patch"].shape == (1, 64, 64, 64)
    assert item["patch"].dtype.is_floating_point
    assert int(item["grade_target"]) == 3


def test_cached_patch_is_float32_and_correct_shape(cached_scan):
    patch = read_cached_patch(os.path.join(cached_scan, "scan1.npy"), (16, 16, 16), (64, 64, 64))

    assert patch.shape == (64, 64, 64)
    assert patch.dtype == np.float32


def test_extract_patch_preserves_source_dtype():
    """Regression: float64 buffers multiplied peak memory by 4 during training."""
    volume = np.zeros((20, 20, 20), dtype=np.float16)
    patch = extract_patch(volume, (10, 10, 10), (8, 8, 8))

    assert patch.dtype == np.float16
    assert patch.shape == (8, 8, 8)


def test_manifest_never_leaves_a_fold_empty():
    """Regression: subset % 5 collapsed the cohort into fold 0 and emptied splits."""
    samples, label_source = prepare_luna16_dataset_manifest(
        data_dir=str(os.path.join(PROJECT_ROOT, "data", "does-not-exist")),
        fallback_dir=str(os.path.join(PROJECT_ROOT, "data", "does-not-exist")),
        lidc_xml_dir=str(os.path.join(PROJECT_ROOT, "data", "no-lidc")),
        num_folds=5,
    )

    assert label_source in ("proxy", "real")
    assert len(samples) > 0
    for fold in range(5):
        assert any(s["fold"] == fold for s in samples), f"fold {fold} has no samples"
    # Every sample must carry fold + positivity metadata used by the trainer
    for s in samples:
        assert isinstance(s["fold"], int)
        assert s["is_positive"] in (0, 1)


def test_manifest_samples_include_volume_path_when_scans_exist(tmp_path):
    scan_dir = tmp_path / "subset0"
    scan_dir.mkdir()
    (scan_dir / "abc.mhd").write_text(
        "DimSize = 4 4 4\nElementSpacing = 1 1 1\nOffset = 0 0 0\n", encoding="utf-8"
    )

    samples, _ = prepare_luna16_dataset_manifest(
        data_dir=str(tmp_path),
        fallback_dir=str(tmp_path),
        lidc_xml_dir=str(tmp_path / "nolidc"),
        num_folds=5,
    )

    assert samples, "discovered volumes should produce candidates"
    assert all(s.get("mhd_path") for s in samples)
    assert all(os.path.exists(s["mhd_path"]) for s in samples)


def test_diameter_proxy_buckets():
    assert diameter_to_proxy_label(3.0) == 1
    assert diameter_to_proxy_label(7.0) == 2
    assert diameter_to_proxy_label(10.0) == 3
    assert diameter_to_proxy_label(18.0) == 4
    assert diameter_to_proxy_label(25.0) == 5


def test_gradcam_slice_renders_on_modern_matplotlib():
    """Regression: matplotlib >= 3.9 removed cm.get_cmap and broke every endpoint."""
    data_url = gradcam3d.generate_mock_ct_gradcam_slice(diameter_mm=12.0, has_nodule=True)

    assert data_url.startswith("data:image/png;base64,")
    assert len(data_url) > 1000


def test_gradcam_blend_handles_flat_cam():
    ct = np.zeros((32, 32), dtype=np.float32)
    cam = np.zeros((32, 32), dtype=np.float32)
    assert gradcam3d.blend_axial_slice_gradcam(ct, cam).startswith("data:image/png;base64,")


def test_uploaded_slice_analysis_grades_by_size():
    from app.slice_analysis import analyze_slice

    def make_slice(radius: int) -> bytes:
        arr = np.full((256, 256), 25, dtype=np.uint8)
        yy, xx = np.ogrid[:256, :256]
        arr[((yy - 128) / 62) ** 2 + ((xx - 128) / 72) ** 2 <= 1] = 85
        if radius > 0:
            arr[((yy - 108) / radius) ** 2 + ((xx - 162) / radius) ** 2 <= 1] = 190
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        return buffer.getvalue()

    small = analyze_slice(make_slice(4))
    large = analyze_slice(make_slice(16))
    clear = analyze_slice(make_slice(0))

    assert small["top_nodule"] is not None
    assert large["top_nodule"]["diameter_mm"] > small["top_nodule"]["diameter_mm"]
    assert large["grade_int"] >= small["grade_int"]
    assert clear["malignancy_score"] == 0
    assert clear["lung_rads"] == "Lung-RADS 1"
    assert clear["top_nodule"] is None
