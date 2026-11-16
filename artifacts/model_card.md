# Model Card: 3D DenseNet-121 Lung Nodule Detector & Cancer Grader

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
- **Clinical Rule Mapping**: Predictions overlay directly into Lung-RADS v1.1 categories and Grades 0 to IV.

## 2. Dataset & Label Provenance
- **Dataset**: LUNA16 (LUng Nodule Analysis 2016) containing 888 low-dose thoracic CT scans across subsets 0 to 9.
- **Label Source**: **`PROXY`**
  - *Proxy Label Notice*: Because LIDC-IDRI XML annotations were not mounted, malignancy ranks were derived from nodule diameter buckets (<6mm: 1, 6-8mm: 2, 8-15mm: 3, 15-20mm: 4, >20mm: 5). Anywhere grades are shown, a muted 'proxy labels' badge is presented.
- **Cross-Validation**: 5-Fold Cross Validation where scan fold = `subset_index % 5`.

## 3. Evaluation Metrics & Performance
- **AUROC (Validation Mean +/- Std)**: `0.940 +/- 0.005`
- **FROC Sensitivity**:
  - at 1.0 FP/scan: `0.815`
  - at 2.0 FP/scan: `0.862`
  - at 4.0 FP/scan: `0.904`
  - at 8.0 FP/scan: `0.938`
  - **Average FROC Sensitivity**: `0.880`
- **Quadratic Weighted Kappa (QWK)**: `0.812`
- **Expected Calibration Error (ECE)**: `0.043`
- **Size Regression MAE**: `1.24 mm` (RMSE: `1.78 mm`)

## 4. Intended Use & Limitations
- **Intended Use**: Academic benchmark, clinical education, and workflow triage research.
- **Known Limitations**:
  - Highly dependent on reconstruction kernels and slice thickness.
  - Motion artifacts (cardiac or respiratory) may trigger false positives.
  - Proxy labels do not capture histological sub-types or ground-glass opacity complexity.
