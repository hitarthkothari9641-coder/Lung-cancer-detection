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
  
  The active label source is recorded in `artifacts/metrics.json` and `artifacts/model_card.md` so any reported result states whether it came from real LIDC reads or diameter proxies.

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

## 3. Quick Start (Windows, one click)

### `start.bat` — launch the application
Double-click **`start.bat`** in the project root. It will:
1. Create a `.venv` Python environment if one does not exist.
2. Install backend Python packages and frontend npm packages (first run only).
3. Generate the sample patient cohort if `data/sample_patients.json` is missing.
4. Start the FastAPI backend on `http://127.0.0.1:8000` and the Next.js UI on `http://localhost:3000`.
5. Open the browser to the app.

Without trained weights the app runs on the built-in demo cohort, so every screen works immediately.

### `train.bat` — train the model
Double-click **`train.bat`** to run the complete two-step pipeline:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `training/preprocess.py` | Resamples every `.mhd` in `data/LUNA16` to 1.0 mm isotropic, clips HU to `[-1000, 400]`, and caches normalized `float16` volumes in `./cache/`. |
| 2 | `training/train.py` | Runs 5-fold cross-validation, saves `artifacts/model.pt` + `artifacts/model_ts.pt`, and writes measured metrics and plots. |

Arguments are forwarded to `train.py`, e.g. `train.bat --epochs 2` or `train.bat --fold 0 --dry-run`.

Both scripts are idempotent — re-running them reuses the existing environment and cache.

### Manual (macOS / Linux)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r training/requirements.txt
cd frontend && npm install && cd ..

python training/preprocess.py     # step 1
python training/train.py          # step 2

uvicorn backend.app.main:app --host 0.0.0.0 --port 8000   # terminal A
cd frontend && npm run dev                                # terminal B
```
Open **[http://localhost:3000](http://localhost:3000)**.

The browser calls same-origin `/api/*`; `next.config.js` proxies those requests to
`BACKEND_URL` (default `http://127.0.0.1:8000`), so no CORS setup is required.

---

## 4. Training on LUNA16 & Full Evaluation

### Dataset layout
```
data/LUNA16/
├── annotations.csv
├── candidates_V2.csv
└── subset0/ ... subset9/
    ├── <seriesuid>.mhd
    └── <seriesuid>.raw
```
Point `data_dir` in `training/config.yaml` at the dataset root. If the CSVs are absent the
discovered volumes are still used as unlabelled screening candidates; if no volumes exist at
all, a small synthetic cohort keeps the pipeline runnable for smoke testing.

### 1. Preprocessing (Step 1)
```bash
python training/preprocess.py --max_scans 50
```

### 2. 5-Fold Cross Validation (Step 2)
```bash
python training/train.py --config training/config.yaml
```
- Folds come from `subset_index % 5` when the full subset layout is present, otherwise from a
  stable per-series hash, so no fold is ever empty.
- AdamW (`lr = 1e-4`), linear warmup, cosine annealing, AMP on CUDA, and early stopping.
- Saves the best checkpoint to `artifacts/model.pt` and exports TorchScript to `artifacts/model_ts.pt`.
- Writes **measured** metrics to `artifacts/metrics.json` plus FROC / reliability / confusion plots.

Useful flags: `--epochs`, `--fold`, `--batch-size`, `--device cpu|cuda`, `--dry-run`.

> On CPU the batch size is automatically capped by `cpu_batch_size` in the config so a
> training step fits in ordinary laptop RAM.

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

---

## 5. Reported Performance

All headline numbers are written by `training/train.py` to `artifacts/metrics.json`
after the run, and each entry carries a `results_source` field:

| Metric | Source | Evaluation standard |
| :--- | :--- | :--- |
| **Validation AUROC (mean &plusmn; std)** | `auroc_mean` / `auroc_std` | 5-fold cross validation |
| **FROC sensitivity @ 1, 2, 4, 8 FP/scan** | `froc.sensitivities` | LUNA16 competition metric |
| **Quadratic Weighted Kappa (QWK)** | `qwk` | Ordinal 5-class agreement |
| **Expected Calibration Error (ECE)** | `ece` | 10-bin reliability diagram |
| **Size regression MAE / RMSE** | `size_mae` / `size_rmse` | Axial diameter error |

- `results_source: "measured"` — computed from the validation splits of your run.
- `results_source: "benchmark_reference"` — a reference curve shown only when no fold
  results exist (for example when running `evaluate.py` on its own). These are **not**
  your model's results.

Scores from a small or synthetic cohort are not meaningful clinically; use the full
LUNA16 release for any publishable comparison.

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
  ├── 4 summary cards (scans reviewed, scans with a nodule, high-risk %, mean confidence)
  ├── Grade distribution (Grades 0 to IV)
  └── Recent scans queue (top 6 patients)

[ Patient Detail / Scan Viewer ]
  ├── Left: axial CT slice with a toggleable lesion highlight
  └── Right: severity grade, malignancy 1..5 meter, Lung-RADS follow-up guidance

[ Scan Upload ]
  ├── Drag-and-drop PNG/JPG image or MetaImage (.mhd/.raw) volume
  └── Sample scan loaders for immediate testing
```

---

## 8. License & Citation

Distributed for academic and research purposes under the MIT License.
If using this repository in research, please cite the LUNA16 and LIDC-IDRI consortium papers:
- *Setio et al., "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge," Medical Image Analysis, 2017.*
- *Armato et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans," Medical Physics, 2011.*
