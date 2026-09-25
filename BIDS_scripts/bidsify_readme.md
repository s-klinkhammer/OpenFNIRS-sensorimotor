# BIDSification & Dataset Processing Scripts

This directory contains the core scripts used to structure, reduce, and convert the raw fNIRS recordings into a standardized **BIDS** (Brain Imaging Data Structure) dataset and **BIDS derivatives**.

---

## 📜 Script Overview

### 1. `reduce_snirf.py` (Montage Reduction)
Reduces the original 8x8 optode montage to the target 5x5 configuration.
* **Functionality:** 
  * Keeps only channels labeled as `RAW`.
  * Removes specific sources (S2, S4, S5) and detectors (D2, D4, D7).
  * Renumbers remaining optodes consecutively (S1–S5, D1–D5).
  * Trims probe arrays (`sourcePos2D/3D`, `detectorPos2D/3D`) and updates `ChannelMask`.
* **Usage:**
  ```bash
  python reduce_snirf.py --input /path/to/raw_8x8 --output /path/to/reduced_5x5
  ```

---

### 2. `bidsify_raw.py` (Raw BIDS Export)
Converts the reduced 5x5 `.snirf` recordings and metadata into a valid raw BIDS dataset structure.
* **Functionality:**
  * Maps recordings to BIDS conventions (`sub-XXX/nirs/sub-XXX_task-motor_run-N_nirs.snirf`).
  * Generates hardware sidecars: `channels.tsv`, `optodes.tsv`, and `coordsystem.json` using spatial mappings from `probeInfo.mat`.
  * Extracts task events (`hand_clenching`, `finger_tapping`) from SNIRF trigger groups and writes `events.tsv`.
  * Writes dataset-level metadata: `README`, `dataset_description.json`, `participants.tsv`, `participants.json`, and `scans.tsv`.
  * Excludes participants specified in the exclusion list (i.e., participants that did not agree to public data sharing).
* **Usage:**
  ```bash
  python bidsify_raw.py \
      --originals /path/to/nirx_exports \
      --snirf /path/to/reduced_5x5 \
      --demographics /path/to/demographics.xlsx \
      --exclude /path/to/exclude.xlsx \
      --output /path/to/bids_root
  ```

---

### 3. `bidsify_derivatives.py` (BIDS Derivatives Export)
Organizes the preprocessed output exported from Satori into the `derivatives/nirs-preproc/` BIDS directory.
* **Functionality:**
  * Structures preprocessed `.snirf` files into `derivatives/nirs-preproc/sub-XXX/nirs/`.
  * Generates BIDS JSON sidecars for each preprocessed recording containing preprocessing details (filter parameters, normalizations, TDDR settings, epoching bounds).
  * Generates derivative-level `README` and `dataset_description.json`.
* **Usage:**
  ```bash
  python bidsify_derivatives.py \
      --source /path/to/satori_exports \
      --bids-root /path/to/bids_root \
      --exclude /path/to/exclude.xlsx
  ```

---

## 🛠️ Requirements

Requirements for executing these scripts:
* Python 3.9+
* `h5py`, `numpy`, `pandas`, `scipy`, `openpyxl`

*(All required packages are included in the main `environment.yml` at the project root.)*