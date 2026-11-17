"""
train.py - 5-Fold Cross Validation Trainer for 3D Lung Nodule Detection and Grading.

Features:
- Deterministic seeding (random, numpy, torch).
- 5-Fold Cross Validation where fold index is derived from the LUNA16 subset layout.
- Multi-task optimization: BCE (detection), CORAL (ordinal grading), SmoothL1 (size).
- AdamW optimizer with configurable warmup and cosine annealing learning rate schedule.
- Automatic Mixed Precision (AMP) when a CUDA device is available.
- Optional TensorBoard logging of losses, AUROC, QWK, and MAE.
- Saves the best fold checkpoint to artifacts/model.pt and exports TorchScript to
  artifacts/model_ts.pt (always traced from a CPU copy so it loads anywhere).
- Computes real validation metrics, then writes artifacts/model_card.md and plots.

Usage:
    python training/train.py                 # full 5-fold training
    python training/train.py --epochs 2      # quick run
    python training/train.py --fold 0 --dry-run
"""

import os
import sys

# Ensure training directory and project root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import contextlib
import copy
import math
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
from sklearn.metrics import roc_auc_score, confusion_matrix

# Import local modules
from model import LungNoduleMultiTaskNet, build_model
from losses import MultiTaskLoss, binary_to_ordinal_prediction
from dataset import Luna16Dataset, prepare_luna16_dataset_manifest
from evaluate import (
    compute_qwk,
    compute_froc,
    compute_calibration_ece,
    generate_evaluation_artifacts,
)

# TensorBoard is optional: training must never fail because it is missing
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolves a config path relative to the project root (never the caller's CWD)."""
    if path is None:
        return None
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(PROJECT_ROOT, expanded))


def load_config(config_path: str = "training/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file, falling back to the bundled config."""
    candidates = [
        config_path,
        resolve_path(config_path) or "",
        os.path.join(SCRIPT_DIR, "config.yaml"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(f"Configuration file not found: {config_path}")


class _NullScaler:
    """Minimal GradScaler stand-in used when AMP is disabled or unavailable."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def make_grad_scaler(device: torch.device, enabled: bool):
    """Version-safe GradScaler (torch.amp.GradScaler only exists in torch>=2.3)."""
    if not enabled:
        return _NullScaler()
    try:
        return torch.amp.GradScaler(device=device.type, enabled=True)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:  # pragma: no cover - very old / CPU-only builds
        return _NullScaler()


def autocast_context(device: torch.device, enabled: bool):
    """Version-safe autocast context manager."""
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover
        try:
            return torch.cuda.amp.autocast(enabled=True)
        except Exception:
            return contextlib.nullcontext()


def write_model_card(
    output_path: str,
    label_source: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Writes a comprehensive, FDA-aligned research prototype model card.
    """
    froc = metrics.get("froc", {}).get("sensitivities", {}) or {}

    def _froc(key: str, default: float = 0.0) -> float:
        try:
            return float(froc.get(key, default))
        except (TypeError, ValueError):
            return default

    proxy_note = ""
    if label_source != "real":
        proxy_note = (
            "- *Proxy Label Notice*: LIDC-IDRI XML annotations were not found, so malignancy "
            "ranks were derived from nodule diameter buckets (<6mm: 1, 6-8mm: 2, 8-15mm: 3, "
            "15-20mm: 4, >20mm: 5).\n"
        )

    card_content = f"""# Model Card: 3D DenseNet-121 Lung Nodule Detector & Cancer Grader

> [!WARNING]
> **Research prototype. Not a medical device. Not for clinical diagnosis.**
> This algorithm has not been evaluated or cleared by the FDA or any medical regulatory authority.
> It must NOT be used as a primary diagnostic tool or substitute for professional radiological evaluation.

## 1. Overview
- **Model Architecture**: 3D DenseNet-121 Multi-Task Backbone with pooled multi-head outputs.
- **Input Dimensions**: 3D axial CT volumes/patches of shape `[1, 64, 64, 64]` at 1.0 mm isotropic spacing.
- **Output Tasks**:
  1. **Nodule Detection**: Continuous logit / sigmoid confidence probability.
  2. **Cancer Grading**: 5-class ordinal malignancy scale (1 to 5) trained via CORAL cumulative-link loss.
  3. **Size Estimation**: Continuous equivalent diameter regression in millimeters (Smooth L1).
- **Clinical Rule Mapping**: Predictions map into Lung-RADS v1.1 categories and Grades 0 to IV.

## 2. Dataset & Label Provenance
- **Dataset**: LUNA16 (LUng Nodule Analysis 2016) low-dose thoracic CT scans.
- **Label Source**: **`{label_source.upper()}`**
{proxy_note}- **Cross-Validation**: 5-Fold Cross Validation across the available scan cohort.

## 3. Evaluation Metrics & Performance
> The numbers below are measured on the validation split of this training run.
> Fold-level values can be noisy when the cohort is small.

- **AUROC (Validation Mean +/- Std)**: `{metrics.get('auroc_mean', 0.0)} +/- {metrics.get('auroc_std', 0.0)}`
- **FROC Sensitivity**:
  - at 1.0 FP/scan: `{_froc('1.0'):.3f}`
  - at 2.0 FP/scan: `{_froc('2.0'):.3f}`
  - at 4.0 FP/scan: `{_froc('4.0'):.3f}`
  - at 8.0 FP/scan: `{_froc('8.0'):.3f}`
  - **Average FROC Sensitivity**: `{metrics.get('froc', {}).get('average_sensitivity', 0.0)}`
- **Quadratic Weighted Kappa (QWK)**: `{metrics.get('qwk', 0.0)}`
- **Expected Calibration Error (ECE)**: `{metrics.get('ece', 0.0)}`
- **Size Regression MAE**: `{metrics.get('size_mae', 0.0)} mm` (RMSE: `{metrics.get('size_rmse', 0.0)} mm`)

## 4. Intended Use & Limitations
- **Intended Use**: Academic benchmark, clinical education, and workflow triage research.
- **Known Limitations**:
  - Highly dependent on reconstruction kernels and slice thickness.
  - Motion artifacts (cardiac or respiratory) may trigger false positives.
  - Proxy labels do not capture histological sub-types or ground-glass opacity complexity.
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"Exported model card to: {output_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _safe_auroc(targets: List[float], probs: List[float]) -> Optional[float]:
    """AUROC that returns None instead of raising for single-class splits."""
    if len(targets) < 2 or len(set(targets)) < 2:
        return None
    try:
        return float(roc_auc_score(targets, probs))
    except Exception:
        return None


def train_single_fold(
    fold_idx: int,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    writer: Optional[Any] = None,
) -> Tuple[LungNoduleMultiTaskNet, Dict[str, Any]]:
    """Trains the model on a single cross-validation fold and returns validation data."""
    batch_size = max(1, int(cfg.get("batch_size", 16)))
    if device.type == "cpu":
        # Full-volume 3D activations are far cheaper on GPU; cap CPU batches so the
        # training step fits comfortably in a laptop's RAM.
        batch_size = max(1, min(batch_size, int(cfg.get("cpu_batch_size", 2))))
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    epochs = max(1, int(cfg.get("epochs", 60)))
    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 3)))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    num_workers = max(0, int(cfg.get("num_workers", 0)))

    patch_size = tuple(int(v) for v in cfg.get("patch_size", [64, 64, 64]))
    cache_dir = resolve_path(cfg.get("cache_dir", "./cache")) or "./cache"

    train_ds = Luna16Dataset(
        train_samples,
        cache_dir=cache_dir,
        patch_size=patch_size,
        is_training=True,
        jitter_voxels=int(cfg.get("jitter_voxels", 4)),
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = Luna16Dataset(
        val_samples,
        cache_dir=cache_dir,
        patch_size=patch_size,
        is_training=False,
        jitter_voxels=0,
        seed=int(cfg.get("seed", 42)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model = build_model(
        in_channels=int(cfg.get("in_channels", 1)),
        num_grading_classes=int(cfg.get("num_grading_classes", 5)),
        backbone=cfg.get("backbone", "densenet121"),
    ).to(device)

    # Positive class weighting for the detection head
    num_pos = sum(1 for s in train_samples if s["is_positive"])
    num_neg = len(train_samples) - num_pos
    pos_weight = float(num_neg / max(1, num_pos)) if num_pos > 0 else 1.0

    criterion = MultiTaskLoss(
        num_classes=int(cfg.get("num_grading_classes", 5)),
        detection_weight=float(cfg.get("detection_weight", 1.0)),
        grading_weight=float(cfg.get("grading_weight", 1.0)),
        size_weight=float(cfg.get("size_weight", 0.5)),
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cosine_epochs = max(1, epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr * 0.01)
    scaler = make_grad_scaler(device, use_amp)

    best_val_auroc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = max(1, int(cfg.get("early_stop_patience", 12)))
    patience_counter = 0

    print(
        f"--- Training Fold {fold_idx}: Train {len(train_samples)} samples, "
        f"Val {len(val_samples)} samples ---"
    )

    for epoch in range(1, epochs + 1):
        # Linear warmup is applied *before* the epoch runs, then cosine decay takes over
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_factor = epoch / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * warmup_factor
        elif epoch == warmup_epochs + 1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            patches = batch["patch"].to(device)
            det_targets = batch["det_target"].to(device)
            grade_targets = batch["grade_target"].to(device)
            size_targets = batch["size_target"].to(device)

            with autocast_context(device, use_amp):
                det_logits, grade_logits, size_pred = model(patches)
                loss, loss_dict = criterion(
                    det_logits, grade_logits, size_pred,
                    det_targets, grade_targets, size_targets,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_dict["loss_total"]
            num_batches += 1

        if warmup_epochs == 0 or epoch > warmup_epochs:
            scheduler.step()

        avg_train_loss = train_loss / max(1, num_batches)

        # ---------------- Validation ----------------
        model.eval()
        val_det_probs: List[float] = []
        val_det_trues: List[float] = []
        val_grade_preds: List[int] = []
        val_grade_trues: List[int] = []
        val_size_preds: List[float] = []
        val_size_trues: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                patches = batch["patch"].to(device)
                det_targets = batch["det_target"].to(device)
                grade_targets = batch["grade_target"].to(device)
                size_targets = batch["size_target"].to(device)

                det_logits, grade_logits, size_pred = model(patches)
                probs = torch.sigmoid(det_logits).cpu().numpy().flatten()
                pred_ranks, _ = binary_to_ordinal_prediction(grade_logits)

                val_det_probs.extend(probs.tolist())
                val_det_trues.extend(det_targets.cpu().numpy().flatten().tolist())

                pos_mask = det_targets.cpu().numpy().flatten() > 0.5
                if np.sum(pos_mask) > 0:
                    val_grade_preds.extend(pred_ranks.cpu().numpy()[pos_mask].tolist())
                    val_grade_trues.extend(grade_targets.cpu().numpy()[pos_mask].tolist())
                    val_size_preds.extend(size_pred.cpu().numpy().flatten()[pos_mask].tolist())
                    val_size_trues.extend(size_targets.cpu().numpy().flatten()[pos_mask].tolist())

        val_auroc = _safe_auroc(val_det_trues, val_det_probs)
        val_qwk = compute_qwk(val_grade_trues, val_grade_preds) if len(val_grade_trues) > 0 else None
        val_mae = (
            float(np.mean(np.abs(np.array(val_size_preds) - np.array(val_size_trues))))
            if len(val_size_trues) > 0
            else None
        )

        if writer is not None:
            writer.add_scalar(f"Loss/Train_Fold{fold_idx}", avg_train_loss, epoch)
            if val_auroc is not None:
                writer.add_scalar(f"Metrics/Val_AUROC_Fold{fold_idx}", val_auroc, epoch)
            if val_qwk is not None:
                writer.add_scalar(f"Metrics/Val_QWK_Fold{fold_idx}", val_qwk, epoch)
            if val_mae is not None:
                writer.add_scalar(f"Metrics/Val_SizeMAE_Fold{fold_idx}", val_mae, epoch)

        auroc_text = f"{val_auroc:.4f}" if val_auroc is not None else "n/a"
        qwk_text = f"{val_qwk:.3f}" if val_qwk is not None else "n/a"
        mae_text = f"{val_mae:.2f} mm" if val_mae is not None else "n/a"
        print(
            f"  Epoch {epoch}/{epochs} | Loss: {avg_train_loss:.4f} | "
            f"Val AUROC: {auroc_text} | Val QWK: {qwk_text} | Size MAE: {mae_text}"
        )

        # Keep the checkpoint with the best validation AUROC (or the latest if undefined)
        if val_auroc is None:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Restore the best weights observed during this fold
    model.load_state_dict(best_state)

    # Final pass with the restored weights so reported metrics match the saved model
    model.eval()
    final_probs: List[float] = []
    final_trues: List[float] = []
    final_grade_preds: List[int] = []
    final_grade_trues: List[int] = []
    final_size_preds: List[float] = []
    final_size_trues: List[float] = []
    series_ids: List[str] = []

    with torch.no_grad():
        for batch in val_loader:
            patches = batch["patch"].to(device)
            det_targets = batch["det_target"].to(device)
            grade_targets = batch["grade_target"].to(device)
            size_targets = batch["size_target"].to(device)

            det_logits, grade_logits, size_pred = model(patches)
            final_probs.extend(torch.sigmoid(det_logits).cpu().numpy().flatten().tolist())
            final_trues.extend(det_targets.cpu().numpy().flatten().tolist())
            pred_ranks, _ = binary_to_ordinal_prediction(grade_logits)
            pos_mask = det_targets.cpu().numpy().flatten() > 0.5
            if np.sum(pos_mask) > 0:
                final_grade_preds.extend(pred_ranks.cpu().numpy()[pos_mask].tolist())
                final_grade_trues.extend(grade_targets.cpu().numpy()[pos_mask].tolist())
                final_size_preds.extend(size_pred.cpu().numpy().flatten()[pos_mask].tolist())
                final_size_trues.extend(size_targets.cpu().numpy().flatten()[pos_mask].tolist())
            series_ids.extend(list(batch["seriesuid"]))

    final_auroc = _safe_auroc(final_trues, final_probs)
    final_qwk = compute_qwk(final_grade_trues, final_grade_preds) if final_grade_trues else None
    final_mae = (
        float(np.mean(np.abs(np.array(final_size_preds) - np.array(final_size_trues))))
        if final_size_trues
        else None
    )
    final_ece, calib_bins = compute_calibration_ece(final_trues, final_probs)
    confusion = (
        confusion_matrix(final_grade_trues, final_grade_preds, labels=[1, 2, 3, 4, 5]).tolist()
        if final_grade_trues
        else []
    )
    froc = compute_froc(
        np.array(final_trues), np.array(final_probs), num_scans=len(set(series_ids)) or 1
    )

    report = {
        "fold": fold_idx,
        "auroc": final_auroc,
        "qwk": final_qwk,
        "ece": final_ece,
        "size_mae": final_mae,
        "size_rmse": (
            float(np.sqrt(np.mean((np.array(final_size_preds) - np.array(final_size_trues)) ** 2)))
            if final_size_trues
            else None
        ),
        "confusion": confusion,
        "calibration": calib_bins,
        "froc": froc,
        "num_val_samples": len(final_trues),
        "num_val_scans": len(set(series_ids)),
        "num_val_positives": int(sum(final_trues)),
    }

    print(
        f"  Fold {fold_idx} complete | AUROC: "
        f"{final_auroc if final_auroc is not None else 'n/a'} | "
        f"QWK: {final_qwk} | MAE: {final_mae}"
    )
    return model, report


def _export_torchscript(model: nn.Module, path: str, patch_size: Tuple[int, int, int]) -> bool:
    """Exports a CPU copy of the model to TorchScript; returns True on success."""
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    dummy = torch.randn(1, 1, *patch_size, device="cpu")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(cpu_model, dummy, strict=False)
            traced = torch.jit.freeze(traced.eval())
            traced.save(path)
        torch.jit.load(path, map_location="cpu")
        print(f"Exported TorchScript model to: {path}")
        return True
    except Exception as trace_error:
        print(f"TorchScript tracing failed ({trace_error}); trying torch.jit.script ...")
    try:
        scripted = torch.jit.script(cpu_model)
        scripted.save(path)
        torch.jit.load(path, map_location="cpu")
        print(f"Exported scripted TorchScript model to: {path}")
        return True
    except Exception as script_error:
        print(f"WARNING: TorchScript export failed ({script_error}).")
        print("         model.pt is still usable by the backend (it loads state_dict).")
        return False


def run_training_pipeline(argv: Optional[List[str]] = None) -> int:
    """Main execution entry point for training 5-fold CV and exporting artifacts."""
    parser = argparse.ArgumentParser(description="Train 3D Lung Nodule Detector and Grader")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to train (0-4), or None for all")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force compute device")
    parser.add_argument("--dry-run", action="store_true", help="1 epoch on fold 0 with a tiny batch")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.dry_run:
        cfg["epochs"] = 1
        cfg["batch_size"] = 2
        if args.fold is None:
            args.fold = 0

    set_seed(int(cfg.get("seed", 42)))

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data_dir = resolve_path(cfg.get("data_dir", "./data/LUNA16")) or "./data/LUNA16"
    fallback_dir = resolve_path(cfg.get("data_dir_fallback", "./")) or PROJECT_ROOT
    lidc_dir = resolve_path(cfg.get("lidc_xml_dir", "./data/LIDC-XML")) or "./data/LIDC-XML"
    cache_dir = resolve_path(cfg.get("cache_dir", "./cache")) or "./cache"
    artifacts_dir = resolve_path(cfg.get("artifacts_dir", "./artifacts")) or "./artifacts"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"Scanning for CT volumes in: {data_dir}")
    samples, label_source = prepare_luna16_dataset_manifest(
        data_dir=data_dir,
        fallback_dir=fallback_dir,
        lidc_xml_dir=lidc_dir,
        neg_pos_ratio=float(cfg.get("neg_pos_ratio", 2.0)),
        seed=int(cfg.get("seed", 42)),
        num_folds=int(cfg.get("num_folds", 5)),
        cache_dir=cache_dir,
    )
    print(f"Total manifest candidates: {len(samples)} | Label Source: {label_source}")

    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=os.path.join(artifacts_dir, "tensorboard"))
        except Exception as exc:
            print(f"TensorBoard logging disabled: {exc}")
    else:
        print("TensorBoard not installed; continuing without training curves.")

    num_folds = int(cfg.get("num_folds", 5))
    folds_to_run = [args.fold] if args.fold is not None else list(range(num_folds))
    fold_reports: List[Dict[str, Any]] = []
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    highest_auroc = -math.inf
    model_cfg = {
        "in_channels": int(cfg.get("in_channels", 1)),
        "num_grading_classes": int(cfg.get("num_grading_classes", 5)),
        "backbone": cfg.get("backbone", "densenet121"),
    }

    for fold_idx in folds_to_run:
        train_samples = [s for s in samples if s["fold"] != fold_idx]
        val_samples = [s for s in samples if s["fold"] == fold_idx]

        if not train_samples or not val_samples:
            print(f"Skipping fold {fold_idx}: empty train/validation split.")
            continue

        model, report = train_single_fold(
            fold_idx, train_samples, val_samples, cfg, device, writer
        )
        fold_reports.append(report)

        score = report["auroc"] if report["auroc"] is not None else -1.0
        if best_state_dict is None or score > highest_auroc:
            highest_auroc = score
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Release fold-local tensors before starting the next fold
        del model
        gc.collect()

    if writer is not None:
        writer.close()

    best_overall_model = build_model(**model_cfg).cpu()
    if best_state_dict is not None:
        best_overall_model.load_state_dict(best_state_dict)
    else:
        print("No fold completed successfully; saving an untrained model for smoke testing.")

    # 1. Save PyTorch weights
    model_pt_path = os.path.join(artifacts_dir, "model.pt")
    torch.save(best_overall_model.state_dict(), model_pt_path)
    print(f"Saved PyTorch weights to: {model_pt_path}")

    # 2. Export TorchScript for high-speed offline serving
    patch_size = tuple(int(v) for v in cfg.get("patch_size", [64, 64, 64]))
    _export_torchscript(best_overall_model, os.path.join(artifacts_dir, "model_ts.pt"), patch_size)

    # 3. Evaluate and generate metrics & plots
    metrics = generate_evaluation_artifacts(
        predictions_per_fold=fold_reports,
        artifacts_dir=artifacts_dir,
        label_source=label_source,
    )

    # 4. Write model card
    write_model_card(os.path.join(artifacts_dir, "model_card.md"), label_source, metrics)

    print("--- Training and Evaluation Pipeline Finished Successfully ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_training_pipeline())
