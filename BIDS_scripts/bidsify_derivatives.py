#!/usr/bin/env python3
"""
Organise the preprocessed fNIRS data of the OpenFNIRS-sensorimotor dataset
as a BIDS derivatives dataset.

The preprocessing itself is done in Satori (the accompanying ``.flow`` file
contains the complete pipeline). This script only copies the exported SNIRF
files into ``<bids_root>/derivatives/nirs-preproc/`` using BIDS file names,
and writes the accompanying metadata (``README``, ``dataset_description.json``
and one JSON sidecar per file).

Input
-----
--source     Folder with the files exported by Satori, containing one
             sub-folder per participant. Only SNIRF files with ``_SD`` in the
             file name are used, and the run letter must be part of the name
             (``RunA`` -> run-1, ``RunB`` -> run-2, ...).
--bids-root  Root of the raw BIDS dataset (created by ``bidsify_raw.py``).
--exclude    Excel file with an ``exclude`` column; every participant with a
             non-empty entry is left out. The ID column must be named
             ``participant_id``, ``id`` or ``subject`` (otherwise the first
             column is used).

Usage
-----
    python bidsify_derivatives.py --source SATORI_EXPORT_DIR \\
        --bids-root BIDS_DIR --exclude exclude.xlsx

Requirements: pandas, openpyxl (see ``environment.yml``).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

# =============================================================================
# 1. METADATA
# =============================================================================

PIPELINE_NAME = "nirs-preproc"

RUN_MAP = {"A": "1", "B": "2", "C": "3", "D": "4"}  # run letter in file name -> BIDS run

PREPROCESSING_DETAILS = (
    "Data were preprocessed in Satori 2.2.4 (Brain Innovation, Maastricht) using the following pipeline: "
    "(a) raw signals of each run were converted to optical density (OD); "
    "(b) OD signals were transformed into hemoglobin concentrations using the modified Beer–Lambert law (mBLL); "
    "(c) motion-related spikes were corrected using monotonic interpolation (iterations = 10, lag = 5 s, threshold = 3.5, influence = 0.5); "
    "(d) temporal derivative distribution repair (TDDR) with restoration of high-frequency components was applied; "
    "(e) GLM-based SCC regression was performed separately for both chromophores; "
    "(f) linear detrending was applied, followed by a Butterworth high-pass filter (second order; 0.01 Hz) and "
    "Gaussian smoothing low-pass filter (0.09 Hz); (g) the data were z-normalized; "
    "(h) the continuous time series were trimmed to 10 s before the first event and 20 s after the last event, "
    "and finally (i) the data were segmented into epochs ranging from -5 s to 25 s relative to event onset. "
    "Given a task duration of 10 s, this epoch additionally included a 15-s post-task period."
)

DATASET_DESCRIPTION = {
    "Name": "PREPROCESSED: fNIRS sensorimotor tasks: Hand clenching & Finger tapping",
    "BIDSVersion": "1.11.1",
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": "Satori",
            "Version": "2.2.4",
            "Description": (
                "Preprocessing pipeline: 1. Raw to Concentration Changes, 2. Spike Removal, "
                "3. TDDR, 4. Short-channel regression, 5. Bandpass Filter (0.01-0.09 Hz) + Detrending, "
                "6. Z-Transformation, 7. Trimming (10s pre first, 20s post last event)."
            ),
        }
    ],
    "SourceDatasets": [
        {"Description": "Raw fNIRS data located in the root BIDS directory"}
    ],
}


def build_sidecar(bids_sub, run):
    """Return the JSON sidecar content for one preprocessed recording."""
    return {
        "Description": "Preprocessed fNIRS data (Concentration Changes) using Satori.",
        "Sources": [f"bids:{bids_sub}/nirs/{bids_sub}_task-motor_run-{run}_nirs.snirf"],
        "Units": "z-score",
        "SoftwareVersions": "Satori 2.2.4",
        "FilteringParameters": {
            "LowPass": 0.09,
            "HighPass": 0.01,
            "Method": "Butterworth (High-Pass) and Gaussian Smoothing (Low-Pass)",
            "Order": 2,
            "Detrending": "linear",
            "SpatialFiltering": "GLM-based short channel regression separately for HbO and HbR",
        },
        "Normalization": "z-transform",
        "MotionCorrection": "TDDR and Spike Removal (Monotonic interpolation)",
        "Epoching": {
            "Start": -5,
            "End": 25,
            "Unit": "s",
            "Description": "Includes 10s task and 15s post-task period",
        },
    }


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================


def digits(text):
    """Return only the digits of a string (e.g. 'P012' -> '012')."""
    return re.sub(r"\D", "", str(text))


def load_excluded_ids(path):
    """Return the digit-only IDs of all participants flagged in the 'exclude' column."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next(
        (c for c in df.columns if c in ("participant_id", "id", "subject")),
        df.columns[0],
    )
    if "exclude" not in df.columns:
        return []
    return [digits(x) for x in df.loc[df["exclude"].notna(), id_col].astype(str)]


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)


# =============================================================================
# 3. MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organise Satori-preprocessed fNIRS data as a BIDS derivatives dataset."
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="folder with the Satori export (one sub-folder per participant)")
    parser.add_argument("--bids-root", type=Path, required=True,
                        help="root of the raw BIDS dataset")
    parser.add_argument("--exclude", type=Path, required=True,
                        help="Excel file with an 'exclude' column")
    return parser.parse_args()


def main():
    args = parse_args()
    excluded_ids = load_excluded_ids(args.exclude)

    dest_root = args.bids_root / "derivatives" / PIPELINE_NAME
    dest_root.mkdir(parents=True, exist_ok=True)

    # Dataset-level files
    (dest_root / "README").write_text(
        "# Preprocessing Description\n\n" + PREPROCESSING_DETAILS, encoding="utf-8"
    )
    write_json(dest_root / "dataset_description.json", DATASET_DESCRIPTION)

    # Participant-level files
    processed, skipped = 0, 0
    for subject_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        subject_id = digits(subject_dir.name)
        if subject_id in excluded_ids:
            skipped += 1
            continue

        bids_sub = f"sub-{subject_id}"
        target_dir = dest_root / bids_sub / "nirs"

        for src in sorted(subject_dir.iterdir()):
            if not (src.name.endswith(".snirf") and "_SD" in src.name):
                continue

            match = re.search(r"Run([A-Z])", src.name)
            run = RUN_MAP.get(match.group(1)) if match else None
            if run is None:
                print(f"WARNING {src.name}: could not determine the run - skipped!")
                continue

            base_name = f"{bids_sub}_task-motor_run-{run}_desc-preproc_nirs"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target_dir / f"{base_name}.snirf")
            write_json(target_dir / f"{base_name}.json", build_sidecar(bids_sub, run))
            processed += 1

    print(f"Export finished. Files: {processed}, excluded participants: {skipped}")


if __name__ == "__main__":
    main()
