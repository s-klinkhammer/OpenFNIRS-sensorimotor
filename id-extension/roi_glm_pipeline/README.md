# ROI GLM analysis: gain from additional ID channels (fNIRS motor task)

Analysis scripts for the single-subject GLM results of a finger-tapping /
hand-clenching fNIRS study. They quantify how many participants show
significant task-related activation, which channel responds best, and how many
participants would have had no significant activation without the additional
increased-density (ID) channels.

Terminology: **SD & ID channels** are available in both the standard-density
and the increased-density montage (CH 2, 6, 16); **ID-only channels** exist
only in the increased-density montage (CH 4, 10, 13). Channel selection,
participant number, run mapping, etc. are defined in `glm_common.py`.

## Files

| File | Purpose |
|---|---|
| `glm_common.py` | Shared constants, command-line options, reading of the GLM result workbooks |
| `alpha_correction_utils.py` | Run combination (Fisher's method with direction check), Bonferroni, M_eff (Li & Ji 2005) |
| `SingleSubjectGLM_Satori.flow` | Satori flow that produces the single-subject GLM results (input of the scripts); see *Satori GLM flow* below |
| `01_summary_per_contrast.py` | One summary workbook per contrast: raw T/p per run, best channel, % significant per channel (uncorrected) |
| `02_significance_per_participant.py` | One workbook per contrast: per-participant significance labels per channel and the "gained via ID channels" analysis (HbO+HbR union, Bonferroni with M_eff; fixed family size 6 as robustness check) |
| `04_generate_paper_results.py` | Results tables and sentences for the paper (`.txt`, and `.docx` if python-docx is installed) |

## Input

One Excel workbook per participant and run in `--input-dir`, named in BIDS
style, e.g. `sub-001_task-motor_run-1_nirs_PPT_GLM_Results.xlsx`
(RunA = `run-1`, RunB = `run-2`). Each workbook needs the sheets
`Oxy GLM Contrast Results` (HbO) and `DeOxy GLM Contrast Results` (HbR) with
one row per channel (`CH 1`, `CH 2`, ...) starting in row 3 and T-/p-value
columns per contrast as defined in `CONTRAST_COLUMNS` in `glm_common.py`.
The input data are not part of this repository.

## Satori GLM flow

The GLM result workbooks are produced in Satori (Brain Innovation) with the
flow `SingleSubjectGLM_Satori.flow`. It consists of two nodes: *fNIRS Load CC*
(data loading) followed by *fNIRS Single-Subject GLM* (`fNIRSIModelGLM`). The
data list in the flow is empty, so the data have to be loaded after opening it.

**The flow has to be run separately for the standard-density (SD) and the
increased-density (ID) data**, i.e. load the SD data, run the flow, then load the
ID data and run it again. Each run yields results for all three contrasts:

| # | Contrast | Weights (predictor 1, predictor 2) |
|---|---|---|
| 1 | `1 > Baseline` (Hand clenching > Baseline) | 1, 0 |
| 2 | `2 > Baseline` (Finger tapping > Baseline) | 0, 1 |
| 3 | `1 > 2` (Hand clenching > Finger tapping) | 1, -1 |

Settings in the GLM node (as stored in the flow):

* Predictors `1` and `2`, two-gamma HRF (time to response peak 6 s, time to
  undershoot peak 16 s, response/undershoot dispersion 1, undershoot ratio 6,
  onset 0, kernel width 32).
* Serial-correlation correction enabled (iterative; max. AR order factor 20,
  max. 20 IRLS iterations).
* Motion confounds, additional physiological confounds and short-channel
  confounds are not added as predictors (`addMotionConfounds`,
  `addAdditionalPhysioConfounds`, `addShortChannelConfound` = false).

The exported results (one workbook per participant and run, sheets
`Oxy GLM Contrast Results` and `DeOxy GLM Contrast Results`) are the input
described below.

## Usage

```bash
pip install -r requirements.txt

python 01_summary_per_contrast.py          --input-dir path/to/GLM_results --output-dir output
python 02_significance_per_participant.py  --input-dir path/to/GLM_results --output-dir output
python 04_generate_paper_results.py        --input-dir path/to/GLM_results --output-dir output
```

Options (all scripts): `--input-dir`, `--output-dir`, `--n-participants`
(default 109). Instead of the flags you can set the environment variables
`GLM_INPUT_DIR` and `GLM_OUTPUT_DIR`. Run all scripts from this folder.

Before using `04_generate_paper_results.py`, check `CHANNEL_LABELS` in that
file: it maps channel numbers to the optode-pair labels printed in the tables
and has to match your montage.

## Methods in brief

* **Run combination.** RunA and RunB p-values are combined per participant and
  channel with Fisher's method. Runs are only combined if their T-values have
  the same sign, and for the "> Baseline" contrasts the shared direction must
  be the expected one (HbO increase, HbR decrease); otherwise the channel
  counts as not significant. `REQUIRE_EXPECTED_DIRECTION` in
  `alpha_correction_utils.py` switches the expected-direction check off.
* **Per-channel significance** (% of participants): Fisher-combined p < 0.05,
  descriptive and uncorrected.
* **Non-responders / gain from ID channels**: per participant, each channel
  family (SD & ID / ID-only) has 6 tests (3 channels x HbO/HbR) and counts as
  significant if any of them is. Bonferroni with the effective number of
  independent tests M_eff (Li & Ji, 2005), estimated once per contrast and
  family from the 6x6 across-participant correlation of the run-averaged
  T-values; a fixed family size of 6 is reported as a robustness check.
* **Best channel**: strongest T in the expected direction (highest T for HbO,
  most negative T for HbR), compared across runs. This is a ranking, not a
  significance test, so no correction applies.

## Reference

Li J, Ji L (2005). Adjusting multiple testing in multilocus analyses using the
eigenvalues of a correlation matrix. *Heredity* 95(3):221-227.
