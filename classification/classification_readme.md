# Classification Pipeline

This directory contains the machine learning pipeline used for feature extraction and classification of fNIRS signal epochs (e.g., Riemannian geometry-based ensemble classification).

---

## 📦 Data Availability

The preprocessed derivatives are hosted on Zenodo (DOI: **TODO — add your Zenodo DOI/link here**) rather than in this repository.

To reproduce the results:

1. Download the derivatives archive from Zenodo.
2. Unzip it into the **repository root** (not into `classification/`) so that the following path exists:
   ```
   OpenFNIRS-sensorimotor/          <- repo root
   ├── data/
   │   └── derivatives/
   │       └── nirs-preproc/
   │           └── sub-001/nirs/*.snirf
   │           └── sub-002/nirs/*.snirf
   │           └── ...
   ├── classification/
   │   ├── generate_npy.py
   │   └── publish_riemann_ensemble.py
   └── ...
   ```
3. Run the scripts below from anywhere — both scripts locate the repo root
   automatically (relative to their own file location), so no path
   arguments or manual configuration are needed.

---

## 📜 Execution Workflow

To reproduce the classification results, execute the scripts in the following order:

### Step 1: Feature Matrix Generation (`generate_npy.py`)
Converts preprocessed `.snirf` time-series data from the BIDS derivatives structure into NumPy array format (`.npy`), extracted as trial epochs based on event markers.

* **Input:** Preprocessed `.snirf` files located in `data/derivatives/nirs-preproc/` (see Data Availability above).
* **Output:** `.npy` feature matrices containing epoched concentration changes ($\Delta\text{HbO}$ / $\Delta\text{HbR}$) per condition and subject, written to `data/sub-XXX/`.
* **Usage:**
  ```bash
  python classification/generate_npy.py
  ```

---

### Step 2: Model Training & Evaluation (`publish_riemann_ensemble.py`)
Trains and evaluates the classification models (e.g., Riemannian covariance matrix estimation, Tangent Space mapping, and ensemble classification across motor tasks).

* **Input:** Generated `.npy` feature files from Step 1.
* **Outputs:** 
  * Classification accuracy and performance metrics (F1-score, confusion matrices).
  * Saved models and evaluation logs exported to the `results/publish_ensemble/` directory.
* **Usage:**
  ```bash
  python classification/publish_riemann_ensemble.py
  ```

---

## 🛠️ Dependencies

Requirements for running the classification pipeline:
* `numpy`
* `scipy`
* `scikit-learn`
* `pyriemann`
* `mne` / `mne-nirs` (for SNIRF loading/epoching)

*(All dependencies are listed in the main `environment.yml` located in the root directory).*