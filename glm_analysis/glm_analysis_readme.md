# Group-Level GLM (Satori Multi-Study GLM)

This folder documents the group-level GLM analysis, run interactively in
**Satori's Multi-Study GLM dialog** (not exportable as a `.flow` workflow,
since it's a click-based interface rather than a node graph).

---

## 📦 Data Availability

This step uses the same `derivatives/nirs-preproc/` folder as the
classification pipeline — see "Getting the data" in the top-level
`README.md` for how to get it in place.

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
6. Satori writes the group-level outputs (`MultiStudy_GLM_Results.txt`,
   `MultiStudy_GLM_Results_MSGLM_Results.xlsx`, and the `*_MSGLM_*.cmp`
   contrast maps) into whichever `sub-XXX/nirs/` folder happens to be
   loaded/open at the time — this is arbitrary and not necessarily
   `sub-001`. Run the cleanup script to collect and relabel them:
   ```bash
   python glm_analysis/move_multistudy_files.py
   ```
   This searches all `sub-*/nirs/` folders for these files and moves them
   into `data/derivatives/nirs-preproc/group/`, replacing the misleading
   `sub-XXX_` prefix on the `.cmp` files with `group_` so they aren't
   mistaken for single-subject results.

## 📁 Files in this directory

* **`generate_msd.py`**: auto-generates the `.msd` run list from the local
  derivatives folder.
* **`move_multistudy_files.py`**: collects the group-level GLM outputs
  Satori scatters into a `sub-XXX/nirs/` folder and moves/relabels them
  into `data/derivatives/nirs-preproc/group/`.
* **`glm_analysis_readme.md`**: this file.
