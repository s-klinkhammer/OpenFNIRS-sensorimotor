# Data Quality Report (Satori + Python)

This folder computes per-channel signal-quality metrics (SNR, CV, cardiac-band
coupling SNR, SCI) used for the data-quality report. Like `preprocessing/`,
part of this runs in **Satori** (v2.2.4) as click-based / `.flow` workflows;
the metrics themselves are then computed in Python from Satori's exports.

---

## 📦 Data Availability

This step starts from the **raw** data, same as `preprocessing/` — see
"Getting the data" in the top-level `README.md`. No separate download: the
raw `sub-XXX/nirs/` folders under `data/` are the same ones used there.

---

## ⚙️ Reproducing this step

There are two independent Satori branches that both start from the same raw
data, followed by one Python script that combines their outputs.

Note: like `preprocessing/`, Satori does not auto-detect paths — you set the
input and output folders manually inside each node's UI, as described below.
`compute_snr_cv.py` uses the same flexible BIDS-folder detection as the
other scripts in this repo: if `data/bids_fnirs_sensorimotor_dataset/`
exists, it looks for `trim_raw`/`trim_od` inside
`data/bids_fnirs_sensorimotor_dataset/derivatives/quality/`; otherwise
directly under `data/derivatives/quality/`. Use whichever of those two
matches your layout as the Save node's output directory below — either
works, but don't mix them (e.g. don't save some subjects to one and some to
the other).

**No batch mode:** confirmed as of Satori v2.2.4, the Load node's file
dialog only lets you select files within a single subject folder, and
there's no way to queue multiple subjects or run a `.flow` workflow across
`sub-XXX` folders automatically. Both branches below have to be repeated
per subject (Load → Run → Save, then move to the next `sub-XXX` folder) —
for N = 100 subjects that's ~100 repetitions per branch.

**Optional:** `../flatten_raw_for_satori.py` (repo root) copies every
subject's raw `.snirf` file into one flat `data/raw_flat/` folder, so the
Load node's file dialog can show and select across all subjects at once
instead of navigating into each `sub-XXX/nirs/` folder — point both
branches' Load nodes at `data/raw_flat/` instead of the nested BIDS folders
if you use it. Test with a handful of subjects first; see the script's
docstring for why.

### Branch A — trimmed raw intensity (`Trim.flow`)
1. Open `quality/Trim.flow` in Satori.
2. In the **Load RAW fNIRS Dataset** node, point it to one subject's raw
   `sub-XXX/nirs/` folder.
3. Run the workflow (Load → Trim: cuts 10 s before the first trigger, 20 s
   after the last — same window as the main preprocessing pipeline).
4. Export/save the trimmed RAW `.snirf` output into:
   ```
   data/derivatives/quality/trim_raw/
   ```
   (or `data/bids_fnirs_sensorimotor_dataset/derivatives/quality/trim_raw/`
   if that's the layout you have — see the note above.)
5. Repeat steps 2–4 for the next subject.

### Branch B — trimmed OD + SCI (`Trim_OD.flow` → `SCI.flow`)
1. Open `quality/Trim_OD.flow` in Satori, point the Load node at one
   subject's raw `sub-XXX/nirs/` folder, and run it (Load → Trim → Raw→OD →
   Save, suffix `_Satori`). Set the Save node's output directory to:
   ```
   data/derivatives/quality/trim_od/
   ```
   (or `data/bids_fnirs_sensorimotor_dataset/derivatives/quality/trim_od/`
   — same note as above.)
2. Open `quality/SCI.flow`, load that subject's trimmed OD output from step
   1, and run it (SCI Channel Rejection, threshold 1.0). This exports one
   `<filename>_OD_rejectedChannels_SCI.txt` per input file, containing the
   per-channel SCI values — save these into the same
   `data/derivatives/quality/trim_od/` folder, alongside the OD files.
3. Repeat steps 1–2 for the next subject.

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
* **`../flatten_raw_for_satori.py`** (repo root): optional helper that copies all subjects' raw files into one flat folder, see the note above.
