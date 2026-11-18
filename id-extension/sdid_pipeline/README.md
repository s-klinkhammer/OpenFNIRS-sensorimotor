# SD vs. ID fNIRS classification pipeline

Code for comparing standard-density (SD) and increased-density (ID) optode
configurations in fNIRS motor-task classification (hand clenching vs. finger
tapping) using Riemannian-geometry pipelines with nested cross-validation.

## Setup
```bash
pip install -r requirements.txt
export SDID_BIDS_ROOT=/path/to/SD_ID_BIDS      # PowerShell: $env:SDID_BIDS_ROOT = "C:\path\to\SD_ID_BIDS"
export SDID_OUTPUT_ROOT=/path/to/output        # optional
```
Input: the preprocessed BIDS derivatives produced by the data-preparation scripts
(`BIDSification_derivatives.py`), i.e. `<root>/ID/derivatives/nirs-preproc/sub-*/nirs/*_desc-preproc_nirs.snirf`
and the same for `SD`. `<root>` is the folder containing the `ID/` and `SD/` BIDS datasets;
see `config.py` for other layouts.

## Run
```bash
python run_all.py --bids-root /path/to/SD_ID_BIDS      # complete pipeline
python run_all.py --list                               # show the steps
```
| Step | Script | Purpose |
|---|---|---|
| 01 | `01_preprocess_data.py` | epoching -> `X.npy` / `y.npy` per participant |
| 02 | `02_nested_cv.py --mode main\|sensitivity` | nested CV classification |
| 03 | `03_stats_report.py` | SD vs. ID (group and individual level) + appendix analyses |
| 04 | `04_group_vs_chance.py` | group/individual level vs. chance (restricted permutation test) |
| 05-07 | `05_..`, `06_..`, `07_..` | signal-quality analysis (optional, needs Satori QC files) |
| 08 | `08_build_results_document.py` | Word document with all results |

Shared code is in `core/` (`cv_core.py`: pipelines and CV; `stats_core.py`: tests and effect sizes).
All settings (alpha, permutations, random seed, channel counts) are in `config.py`.
