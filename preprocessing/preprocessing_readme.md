# fNIRS Preprocessing Pipeline (Satori Workflow)

This folder contains the complete fNIRS preprocessing workflow used to transform raw data into preprocessed data (BIDS derivatives). The pipeline was built and executed in **Satori** (v2.2.4).

---

## 📦 Data Availability

This workflow starts from the **raw** recordings, which are part of the same
BIDS dataset as the derivatives — see "Getting the data" in the top-level
`README.md`. You don't need a separate download: once you've unzipped the
dataset into `data/` as described there, the raw `sub-XXX/nirs/` folders are
already what this workflow reads from.

To reproduce this step:

1. Make sure the dataset is unzipped into `data/` at the repository root
   (see "Getting the data" in the top-level `README.md` if you haven't done
   this yet).
2. Open `satori_preprocessing_workflow.flow` in Satori (v2.2.4).
3. In the **Load RAW fNIRS Dataset** node, point it to one subject's raw
   `sub-XXX/nirs/` folder at a time and select that subject's run files
   (Satori's file dialog is scoped to a single folder, so you can't select
   across multiple `sub-XXX` folders at once).
4. In the **Save Dataset CC** node, set the output directory to
   `data/derivatives/nirs-preproc/` (or the matching path under
   `bids_fnirs_sensorimotor_dataset/`) at the repository root — this is
   already where the dataset's own derivatives live, so this step is only
   needed if you want to regenerate them yourself rather than use the ones
   that shipped with the download.
5. Run the workflow, then repeat steps 3–4 for the next subject.

Note: unlike the classification scripts, Satori does not auto-detect paths,
and — confirmed as of Satori v2.2.4 — there is **no batch/queue option** to
run a `.flow` workflow across multiple subject folders automatically. Each
subject has to be loaded and run individually through the GUI; for the full
N = 100 dataset this means repeating steps 3–5 about 100 times. (This is
different from the Multi-Study GLM step in `glm_analysis/`, where
`generate_msd.py` sidesteps exactly this limitation by pre-building a run
list Satori can import in one go — that trick doesn't apply here since
`.flow` workflows and the Multi-Study GLM dialog are separate parts of
Satori with different import mechanisms.)

**Optional:** `flatten_raw_for_satori.py` (repo root) copies every subject's
raw `.snirf` file into one flat `data/raw_flat/` folder, so the Load node's
file dialog can show and select across all subjects at once instead of
navigating into each `sub-XXX/nirs/` folder:
```bash
python flatten_raw_for_satori.py
```
This does *not* confirm that Satori will actually run the workflow
separately per selected file — test it on a handful of subjects first. If
it doesn't, you've still saved yourself the folder navigation.

---

## 📐 Preprocessing Workflow Overview

The raw intensity fNIRS data undergoes conversion to optical density (OD), concentration changes via the Modified Beer-Lambert Law (MBLL), motion artifact correction, systemic noise regression, frequency filtering, normalization, and data trimming.

![Satori Preprocessing Graph](satori_preprocessing_graph.png)

---

## ⚙️ Step-by-Step Processing Pipeline

### 1. Data Import & Conversion
* **Load RAW fNIRS Dataset (`fNIRSLoadRAWModel`):** Imports raw light intensity data.
* **Raw to Optical Density (`fNIRSIORawToODModel`):** Converts raw intensity values to optical density changes ($\Delta \text{OD}$).
* **OD to Concentration (`fNIRSIOODToConcentrationModel`):** Applies the Modified Beer-Lambert Law (MBLL) to compute oxygenated ($\Delta[\text{HbO}]$) and deoxygenated ($\Delta[\text{HbR}]$) hemoglobin concentration changes.

### 2. Motion Artifact Correction & Noise Reduction
* **Spike Removal (`fNIRSSpikeRemovalCCModel`):**
  * **Threshold:** $3.5$
  * **Lag:** $5.0\text{ s}$ | **Influence:** $0.50$ | **Iterations:** $10$
  * **Interpolation:** Monotonic cubic interpolation
* **TDDR Motion Correction (`fNIRSTDDRCCModel`):** Applies Temporal Derivative Distribution Repair (TDDR) to eliminate baseline shifts and movement spikes while preserving high frequencies (`restoreHF = true`).

### 3. Nuisance Signal & Physiological Filtering
* **Short Channel Regression (`fNIRSIOGLMSSRModel`):** Removes superficial scalp hemodynamics using all available short-separation channels (GLM-SSR; $1$ PCA component) applied to both Oxy ($\text{HbO}$) and Deoxy ($\text{HbR}$) chromophores.
* **Temporal Filtering (`fNIRSIOFilteringModel`):**
  * **High-Pass Filter:** Butterworth at $0.01\text{ Hz}$
  * **Low-Pass Filter:** Gaussian smoothing / Butterworth at $0.09\text{ Hz}$
  * **Linear Detrending:** Enabled

### 4. Normalization & Trimming
* **Normalization (`fNIRSIONormalizationModel`):** Applies $z$-standardization ($z$-transform) across time channels.
* **Data Trimming (`fNIRSTRIMDataCCModel`):**
  * Trims $10.0\text{ s}$ before the first trigger event.
  * Trims $20.0\text{ s}$ after the last trigger event.

### 5. Output Export
* **Save Dataset (`fNIRSCCSaveModel`):** Exports preprocessed concentration data (`.snirf` / derivatives) appending the `_PPT` suffix.

---

## 📁 Files in this Directory

* **`satori_preprocessing_graph.png`**: Visual diagram of the Satori processing pipeline.
* **`satori_preprocessing_workflow.flow`**: The exact Satori workflow .flow file to reproduce these preprocessing steps in Satori.
* **`../flatten_raw_for_satori.py`** (repo root): optional helper that copies all subjects' raw files into one flat folder, see the note above.