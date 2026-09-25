# Data Quality Report (Satori + Python)

This folder computes per-channel signal-quality metrics (SNR, CV, cardiac-band
coupling SNR, SCI) used for the data-quality report. Like `preprocessing/`,
part of this runs in **Satori** (v2.2.4) as click-based / `.flow` workflows;
the metrics themselves are then computed in Python from Satori's exports.

---

## 📦 Data Availability

This step starts from the **raw** data, same as `preprocessing/` — see
`preprocessing/preprocessing_readme.md` for where to get it from Zenodo.

---

## ⚙️ Reproducing this step

There are two independent Satori branches that both start from the same raw
data, followed by one Python script that combines their outputs.

Note: like `preprocessing/`, Satori does not auto-detect paths — you set the
input and output folders manually inside each node's UI, as described below.
Using the exact folder names below means `compute_snr_cv.py` will find
everything without any path arguments afterwards.

### Branch A — trimmed raw intensity (`Trim.flow`)
1. Open `quality/Trim.flow` in Satori.
2. In the **Load RAW fNIRS Dataset** node, point it to your raw data folder.
3. Run the workflow (Load → Trim: cuts 10 s before the first trigger, 20 s
   after the last — same window as the main preprocessing pipeline).
4. Export/save the trimmed RAW `.snirf` output into:
   ```
   data/derivatives/quality/trim_raw/
   ```

### Branch B — trimmed OD + SCI (`Trim_OD.flow` → `SCI.flow`)
1. Open `quality/Trim_OD.flow` in Satori, point the Load node at the same raw
   data folder, and run it (Load → Trim → Raw→OD → Save, suffix `_Satori`).
   Set the Save node's output directory to:
   ```
   data/derivatives/quality/trim_od/
   ```
2. Open `quality/SCI.flow`, load the trimmed OD output from step 1, and run
   it (SCI Channel Rejection, threshold 1.0). This exports one
   `<filename>_OD_rejectedChannels_SCI.txt` per input file, containing the
   per-channel SCI values — save these into the same
   `data/derivatives/quality/trim_od/` folder, alongside the OD files.

### Combine into quality metrics (`compute_snr_cv.py`)
Once both branches have been run for all subjects:
```bash
python quality/compute_snr_cv.py
```
This reads `data/derivatives/quality/trim_raw/` (SNR/CV directly from the
trimmed intensity signal) together with the matching SCI `.txt` files in
`data/derivatives/quality/trim_od/` (SCI + cardiac-band coupling SNR per
source-detector pair), classifies each channel/pair as `good` / `usable` /
`poor`, and writes:

* `results/quality/fNIRSQA_output.csv` — channel-level SNR/CV, one row per
  channel per file
* `results/quality/all_runs_coupling_metrics.csv` — pair-level SCI/cSNR and
  the `keep_suggestion` column

Non-default folders (e.g. if you keep the Satori exports elsewhere) can be
passed explicitly:
```bash
python quality/compute_snr_cv.py --snirf-dir <path> --sci-dir <path> --out-dir <path>
```

**Quality thresholds used**
| Metric | strong/good | usable | poor |
|---|---|---|---|
| SCI | ≥ 0.75 | ≥ 0.50 | < 0.50 |
| cardiac-band coupling SNR | ≥ 13 dB | ≥ 2.5 dB | < 2.5 dB |

A channel is flagged `keep` if SCI is at least usable *and* coupling SNR is
at least usable; otherwise `review_or_reject`.

---

## 📁 Files in this directory

* **`Trim.flow`**: trims raw intensity data to the task window (Branch A).
* **`Trim_OD.flow`**: trims raw data and converts to OD (Branch B, step 1).
* **`SCI.flow`**: computes per-channel SCI from the trimmed OD data (Branch B, step 2).
* **`snirf_io.py`**: minimal h5py-based SNIRF reader shared by `compute_snr_cv.py`
  (deliberately doesn't depend on `snirf`/`mne-nirs` — see its docstring).
* **`compute_snr_cv.py`**: combines both branches into the final quality CSVs.
* **`quality_readme.md`**: this file.
