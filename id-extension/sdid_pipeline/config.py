"""
config.py
---------
Central project configuration. No paths are hard-coded; they are set via
environment variables (or command-line arguments in the scripts that offer them):

    SDID_BIDS_ROOT    Folder containing the BIDS datasets. Expected layout
                      (BIDS_LAYOUT = "separate"):
                        <root>/ID/sub-*/nirs/*_nirs.snirf
                        <root>/SD/sub-*/nirs/*_nirs.snirf
                      or a single dataset (BIDS_LAYOUT = "single", see below).
    SDID_OUTPUT_ROOT  (optional) Output folder.
                      Default: <SDID_BIDS_ROOT>/derivatives/sdid_pipeline

Examples
    Windows PowerShell:  $env:SDID_BIDS_ROOT = "C:\\path\\to\\SD_ID_BIDS"
    macOS / Linux:       export SDID_BIDS_ROOT=/path/to/SD_ID_BIDS
"""

import os
import argparse
from pathlib import Path

# ---- Paths ----
_root = os.environ.get("SDID_BIDS_ROOT")
BIDS_ROOT = Path(_root) if _root else None

_out = os.environ.get("SDID_OUTPUT_ROOT")
if _out:
    CUSTOM_OUTPUT_ROOT = Path(_out)
elif BIDS_ROOT is not None:
    CUSTOM_OUTPUT_ROOT = BIDS_ROOT / "derivatives" / "sdid_pipeline"
else:
    CUSTOM_OUTPUT_ROOT = Path.cwd() / "sdid_output"   # fallback so that importing never fails

PROCESSED_DIR = CUSTOM_OUTPUT_ROOT / "02_processed"    # X.npy / y.npy per participant
RESULTS_MAIN_DIR = CUSTOM_OUTPUT_ROOT / "main"
RESULTS_SENSITIVITY_DIR = CUSTOM_OUTPUT_ROOT / "sensitivity"
RESULTS_QUALITY_DIR = CUSTOM_OUTPUT_ROOT / "quality"

# ---- BIDS layout ----
# "separate": one BIDS dataset per configuration: <root>/<COND>/sub-*/nirs/...
# "single"  : one dataset; the configuration is part of the file name
#             (e.g. ..._acq-SD_...), see BIDS_COND_ENTITY.
BIDS_LAYOUT = "separate"
BIDS_COND_ENTITY = "acq"                       # only for "single": acq / task / ...
BIDS_COND_FOLDERS = {"SD": "SD", "ID": "ID"}   # folder per condition for "separate"

# ---- Preprocessed data (BIDS derivatives) ----
# The analysis reads the preprocessed SNIRF files from
#   <BIDS_ROOT>/<COND>/derivatives/<DERIVATIVES_PIPELINE>/sub-*/nirs/*_nirs.snirf
# These derivatives are produced by the separate data-preparation scripts
# (BIDSification_derivatives.py); this pipeline does not create them.
DERIVATIVES_PIPELINE = "nirs-preproc"

# ---- Statistical parameters ----
ALPHA = 0.05
N_PERM = 5000            # post-hoc permutation tests (SD vs. ID, 03_stats_report.py)
N_PERM_RESTRICTED = 500  # restricted permutation tests vs. chance (04_group_vs_chance.py)
RANDOM_STATE = 42
FDR_METHOD = "fdr_bh"
CHANCE_LEVEL = 0.5
CONDITIONS = ("SD", "ID")

# Channels per chromophore and configuration. Needed by the block_covariances
# pipeline (2 blocks: HbO | HbR) and cross-checked during epoching.
BLOCK_SIZE_BY_CONDITION = {"SD": 13, "ID": 20}


def ensure_output_dirs():
    for d in (PROCESSED_DIR, RESULTS_MAIN_DIR, RESULTS_SENSITIVITY_DIR, RESULTS_QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require_bids_root():
    if BIDS_ROOT is None or not BIDS_ROOT.exists():
        raise SystemExit(
            "SDID_BIDS_ROOT is not set or does not exist.\n"
            "Set the environment variable to the folder containing the BIDS data, e.g.\n"
            "  PowerShell: $env:SDID_BIDS_ROOT = 'C:\\path\\to\\SD_ID_BIDS'\n"
            "  bash:       export SDID_BIDS_ROOT=/path/to/SD_ID_BIDS"
        )


def resolve_mode(description=None):
    """Read --mode main|sensitivity from the command line (prompts in the terminal if omitted)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["main", "sensitivity"], help="analysis mode")
    args, _ = parser.parse_known_args()
    mode = args.mode
    while mode not in ("main", "sensitivity"):
        mode = input("Which analysis? [main/sensitivity]: ").strip().lower()
    return mode


def get_mode_paths(mode):
    """mode: "main" or "sensitivity" -> input/output folders (as str) and file prefix."""
    ensure_output_dirs()
    if mode == "main":
        return {"data_path": str(PROCESSED_DIR), "output_dir": str(RESULTS_MAIN_DIR), "file_prefix": "participant"}
    elif mode == "sensitivity":
        return {"data_path": str(PROCESSED_DIR), "output_dir": str(RESULTS_SENSITIVITY_DIR), "file_prefix": "sensitivity"}
    raise ValueError(f"Unknown mode: {mode!r} (expected 'main' or 'sensitivity')")


def find_latest(directory, contains, suffix=".csv"):
    """Newest file in `directory` whose name contains `contains` (timestamps in the
    names make alphabetical order chronological)."""
    directory = Path(directory)
    candidates = sorted(f.name for f in directory.iterdir()
                        if contains in f.name and f.suffix == suffix)
    if not candidates:
        raise FileNotFoundError(f"No file containing '{contains}' found in {directory}. "
                                f"Has the previous step been run?")
    return str(directory / candidates[-1])
