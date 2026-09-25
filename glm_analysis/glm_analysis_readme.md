# Group-Level GLM (Satori Multi-Study GLM)

This folder documents the group-level GLM analysis, run interactively in
**Satori's Multi-Study GLM dialog** (not exportable as a `.flow` workflow,
since it's a click-based interface rather than a node graph).

---

## 📦 Data Availability

This step takes the same preprocessed derivatives as the classification
pipeline. If you haven't already, download them from Zenodo (DOI: **TODO —
add your Zenodo DOI/link here**) and unzip them into `data/derivatives/nirs-preproc/`
at the repository root — see `classification/classification_readme.md` for
the exact layout.

---

## ⚙️ Reproducing this step

1. Run the helper script to auto-generate the list of runs (avoids adding
   ~200 files one by one via "Add Runs..."):
   ```bash
   python glm_analysis/generate_msd.py
   ```
   This writes `data/group_glm_runs.msd` (not committed — it contains
   machine-specific absolute paths).
2. Open Satori -> **Multi-Study GLM**.
3. **Studies tab** -> "Load .MSD..." -> select `data/group_glm_runs.msd`.
4. **GLM / Contrasts tab** -> define the contrasts:
   - `1 > Baseline` (Hand Clenching > Baseline)
   - `2 > Baseline` (Finger Tapping > Baseline)
   - `1 > 2` (Hand Clenching > Finger Tapping)
   Settings used: "Separate subject predictors" enabled, "Correct serial
   corr." disabled.
5. Click **"GO"**.

## 📁 Files in this directory

* **`generate_msd.py`**: auto-generates the `.msd` run list from the local
  derivatives folder.
* **`glm_analysis_readme.md`**: this file.
