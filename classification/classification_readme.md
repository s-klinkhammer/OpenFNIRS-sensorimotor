# Classification Pipeline

This directory contains the machine learning pipeline used for feature extraction and classification of fNIRS signal epochs (e.g., Riemannian geometry-based ensemble classification).

---

## 📜 Execution Workflow

To reproduce the classification results, execute the scripts in the following order:

### Step 1: Feature Matrix Generation (`generate_npy.py`)
Converts preprocessed `.snirf` time-series data from the BIDS derivatives structure into NumPy array format (`.npy`), extracted as trial epochs based on event markers.

* **Input:** Preprocessed `.snirf` files located in `derivatives/nirs-preproc/`.
* **Output:** `.npy` feature matrices containing epoched concentration changes ($\Delta\text{HbO}$ / $\Delta\text{HbR}$) per condition and subject.
* **Usage:**
  ```bash
  python classification/generate_npy.py \
      --bids-dir ../data/bids_root \
      --output-dir ../data/features
  ```

---

### Step 2: Model Training & Evaluation (`publish_riemann_ensemble.py`)
Trains and evaluates the classification models (e.g., Riemannian covariance matrix estimation, Tangent Space mapping, and ensemble classification across motor tasks).

* **Input:** Generated `.npy` feature files.
* **Outputs:** 
  * Classification accuracy and performance metrics (F1-score, confusion matrices).
  * Saved models and evaluation logs exported to the `results/` directory.
* **Usage:**
  ```bash
  python classification/publish_riemann_ensemble.py \
      --data-dir ../data/features \
      --results-dir ../results
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