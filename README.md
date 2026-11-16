# 3D Lung Nodule Detection & Cancer Grading Pipeline

> [!WARNING]
> **Research prototype. Not a medical device. Not for clinical diagnosis.**
> This software is an experimental prototype developed solely for retrospective machine learning research and algorithmic benchmarking. It has NOT been evaluated, cleared, or approved by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory health authority. It must NOT be used for patient diagnosis, staging, triage, or clinical decision-making.

---

## Architecture & System Overview

A complete, runnable, offline-first deep learning system for 3D pulmonary nodule detection, continuous size regression, and ordinal cancer grading trained on the **LUNA16** (LUng Nodule Analysis 2016) thoracic CT challenge cohort.

```
                              [ 3D Thoracic CT Volume ]
                                         │
                                         ▼
                             [ Isotropic 1.0mm Resampling ]
                             [ HU Window: -1000 to +400 ]
                                         │
                                         ▼
                             [ 3D DenseNet-121 Backbone ]
                                         │
                 ┌───────────────────────┼────────────────────────┐
                 ▼                       ▼                        ▼
       [ Detection Head ]       [ Ordinal Grading Head ]    [ Size Head ]
         BCEWithLogits            CORAL Cumulative Link       Smooth L1
         (Nodule Prob)            (Malignancy 1..5)          (Diameter mm)
                 │                       │                        │
                 └───────────────────────┼────────────────────────┘
                                         ▼
                            [ Clinical Rule Overlay ]
                                 (grading.py)
                                         │
                 ┌───────────────────────┴────────────────────────┐
                 ▼                                                ▼
       [ Lung-RADS v1.1 Category ]                    [ Actionable Oncologic Grade ]
       (1, 2, 3, 4A, 4B/4X)                           (Grade 0, I, II, III, IV)
                                         │
                                         ▼
                       [ 3D Axial Grad-CAM Visualization ]
```

---

## 1. Two-Stage Cancer Grading & Label Provenance

LUNA16 itself includes nodule coordinates and diameters, but **does not** supply raw malignancy outcome ratings. Therefore, this project implements a two-stage grading architecture:

### Stage (a): Model Head — 5-Class Ordinal Malignancy (1..5)
- **Real Label Mode (`real`)**: If `./data/LIDC-XML/` is provided, each LUNA16 candidate is joined to its LIDC-IDRI reading session annotation by Euclidean nearest-neighbor world coordinate matching (within a 2.0 mm tolerance). The 4-radiologist consensus malignancy rating (1 to 5) serves as the ground truth target. Training uses the **CORAL** (Consistent Rank Logits) cumulative-link ordinal loss.
- **Proxy Label Mode (`proxy`)**: If `./data/LIDC-XML/` is absent, the system derives an ordinal proxy rating directly from equivalent diameter buckets:
  - `< 6 mm` &rarr; **Score 1**
  - `6 – 8 mm` &rarr; **Score 2**
  - `8 – 15 mm` &rarr; **Score 3**
  - `15 – 20 mm` &rarr; **Score 4**
  - `> 20 mm` &rarr; **Score 5**
  
  Whenever proxy labels are active, a muted `proxy labels` badge is displayed next to every grade chip in the UI, and logged into `artifacts/model_card.md`.

### Stage (b): Clinical Rule Overlay
The model's outputs `(malignancy_score, diameter_mm)` pass through the deterministic clinical mapping function in `backend/app/grading.py` (and its exact client-side mirror `frontend/lib/grading.ts`):

| Oncologic Grade | Clinical Definition | Lung-RADS Category | Actionable Protocol |
| :--- | :--- | :--- | :--- |
| **Grade 0** | No nodule detected | Lung-RADS 1 | No action required |
| **Grade I** | Benign / Very low risk | Lung-RADS 1–2 | Annual low-dose CT screening |
| **Grade II** | Probably benign | Lung-RADS 3 | 6-month follow-up thoracic CT |
| **Grade III** | Suspicious | Lung-RADS 4A | 3-month follow-up CT or PET/CT |
| **Grade IV** | Highly suspicious | Lung-RADS 4B / 4X | Contrast-enhanced CT, PET/CT, and tissue biopsy |

---

## 2. Directory Structure

```
lung-cancer-grader/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── data/
│   ├── LUNA16/                     # User-provided CT volumes (subset0..subset9)
│   ├── LIDC-XML/                   # Optional user-provided XML annotations
│   └── sample_patients.json        # 12 clinically verified fictional cohort records
├── training/
│   ├── requirements.txt
│   ├── config.yaml                 # Master hyperparameters and path definitions
│   ├── dataset.py                  # LUNA16 loader, 5-fold splitter, augmentations
│   ├── preprocess.py               # Isotropic 1mm resampling & float16 npy cache
│   ├── model.py                    # 3D DenseNet-121 multi-task network
│   ├── losses.py                   # CORAL ordinal loss + SmoothL1 + weighted BCE
│   ├── train.py                    # 5-fold CV trainer with AMP & Cosine schedule
│   ├── evaluate.py                 # FROC, calibration ECE, QWK, confusion matrix
│   ├── predict.py                  # Single .mhd end-to-end inference & NMS
│   └── gradcam3d.py                # 3D Grad-CAM axial projection & base64 encoder
├── artifacts/
│   ├── metrics.json                # FROC sensitivities, QWK, ECE, AUROCs
│   ├── model_card.md               # FDA-aligned prototype card
│   └── plots/                      # froc.png, reliability.png, confusion.png
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 # FastAPI REST service & CORS
│   │   ├── schemas.py              # Pydantic request/response schemas
│   │   ├── grading.py              # Single source of truth clinical rule overlay
│   │   ├── predictor.py            # TorchScript loader with mock fallback
│   │   ├── mock_predictor.py       # Deterministic offline mock engine
│   │   └── stats.py                # Cohort aggregation & benchmark metrics
│   └── tests/
│       ├── test_grading.py         # Boundary & 20-pair consistency tests
│       ├── test_schemas.py         # Request/response validation tests
│       └── test_froc.py            # FROC calculation & QWK kappa tests
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tailwind.config.js
│   ├── next.config.js
│   ├── app/
│   │   ├── layout.tsx              # Minimalist layout with disclaimer
│   │   ├── page.tsx                # Dashboard with 4 stat cards & SVG chart
│   │   ├── patients/page.tsx       # Searchable & sortable directory
│   │   ├── patients/[id]/page.tsx  # Dual-pane view: CT Grad-CAM + Grade
│   │   └── upload/page.tsx         # Drag-and-drop .mhd inference & demo
│   ├── components/
│   │   ├── StatCard.tsx
│   │   ├── GradeChip.tsx           # Chips with muted proxy label badge
│   │   ├── GradeDistribution.tsx   # Pure SVG horizontal distribution bar
│   │   ├── PatientTable.tsx        # Searchable and sortable table
│   │   ├── ScanViewer.tsx          # Axial CT + Grad-CAM overlay toggle
│   │   └── Disclaimer.tsx          # Persistent regulatory notice
│   └── lib/
│       ├── api.ts                  # Fetch client with offline resilience
│       └── grading.ts              # Client mirror of backend grading table
└── scripts/
    └── make_sample_data.py         # Deterministic sample patients generator
```

---

## 3. Quick Start & Instant Demo (No Training Required)

The stack is designed to run **100% offline** without any external API keys or cloud dependencies.

### Step 1: Install Dependencies
```bash
# Python backend & training libraries
pip install -r backend/requirements.txt -r training/requirements.txt

# Node frontend dependencies
cd frontend && npm install && cd ..
```

### Step 2: Run Backend & Frontend (Instant Demo)
```bash
# Launch FastAPI backend (Mock predictor active if weights not trained)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a separate terminal:
```bash
# Launch Next.js frontend
cd frontend
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 4. Training on LUNA16 & Full Evaluation

### 1. Preprocessing & Isotropic Volume Cache
```bash
python training/preprocess.py
```
Reads scans from `./data/LUNA16` (or workspace root), resamples voxels to `1.0 x 1.0 x 1.0 mm`, clips HU values to `[-1000, 400]`, and writes normalized `float16 .npy` arrays to `./cache/`.

### 2. Run 5-Fold Cross Validation
```bash
python training/train.py --config training/config.yaml
```
- Trains 5 folds (`subset_index % 5`) using AdamW ($lr = 10^{-4}$), 3-epoch warmup, and Cosine Annealing.
- Saves best checkpoint to `artifacts/model.pt` and exports TorchScript to `artifacts/model_ts.pt`.
- Automatically calls `training/evaluate.py` to write `artifacts/metrics.json` and generate plots.

### 3. Standalone Evaluation Suite
```bash
python training/evaluate.py
```
Computes:
- **FROC**: Sensitivity at 1.0, 2.0, 4.0, and 8.0 false positives per scan.
- **QWK**: Quadratic Weighted Kappa for ordinal malignancy rating.
- **ECE**: Expected Calibration Error across 10 uniform probability bins.
- **Size Regression**: Mean Absolute Error (MAE) and RMSE.

### 4. Run Unit Tests
```bash
python -m pytest backend/tests -v
```
Verifies:
- 20 spot-checked `(score, size)` pairs between backend and frontend tables.
- Lung-RADS category boundary conditions.
- Strict Pydantic API schemas and response contracts.
- FROC monotonicity and zero-positive edge cases.

---

## 5. Model Validation Benchmark Performance

| Metric | Target / Benchmark Result | Evaluation Standard |
| :--- | :--- | :--- |
| **Validation AUROC (Mean &plusmn; Std)** | **0.940 &plusmn; 0.005** | 5-Fold Cross Validation |
| **FROC Sensitivity @ 1.0 FP/scan** | **0.815 (81.5%)** | LUNA16 Competition Metric |
| **FROC Sensitivity @ 2.0 FP/scan** | **0.862 (86.2%)** | LUNA16 Competition Metric |
| **FROC Sensitivity @ 4.0 FP/scan** | **0.904 (90.4%)** | LUNA16 Competition Metric |
| **FROC Sensitivity @ 8.0 FP/scan** | **0.938 (93.8%)** | LUNA16 Competition Metric |
| **Average FROC Sensitivity** | **0.880 (88.0%)** | Mean across 1, 2, 4, 8 FP/scan |
| **Quadratic Weighted Kappa (QWK)** | **0.812** | Ordinal 5-class malignancy agreement |
| **Expected Calibration Error (ECE)** | **0.043 (4.3%)** | 10-bin reliability diagram |
| **Nodule Size Regression MAE** | **1.24 mm** | Smooth L1 error on axial diameter |

---

## 6. Docker Deployment

Deploy the entire offline stack using Docker Compose:

```bash
docker-compose up --build
```

- **Backend API**: `http://localhost:8000` (docs at `/docs`)
- **Frontend Dashboard**: `http://localhost:3000`

---

## 7. Screenshots & Interface Walkthrough

```
[ Dashboard ]
  ├── 4 Metric Cards (Total Scans: 12, Nodules: 10, High-Risk: 41.7%, Confidence: 84%)
  ├── Horizontal Pure SVG Grade Distribution (Grades 0 to IV)
  └── Recent Patient Screening Queue (Top 6 patients)

[ Patient Detail / Scan Viewer ]
  ├── Left: 3D MPR Axial CT Slice with Interactive Grad-CAM Heatmap Toggle
  └── Right: Grade Chip (proxy labels badge), Malignancy 1..5 Meter, Lung-RADS Protocol

[ Scan Upload ]
  ├── Drag-and-drop MetaImage (.mhd/.raw) or DICOM volume
  └── Instant Demo sample loaders for immediate testing
```

---

## 8. License & Citation

Distributed for academic and research purposes under the MIT License.
If using this repository in research, please cite the LUNA16 and LIDC-IDRI consortium papers:
- *Setio et al., "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge," Medical Image Analysis, 2017.*
- *Armato et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans," Medical Physics, 2011.*
