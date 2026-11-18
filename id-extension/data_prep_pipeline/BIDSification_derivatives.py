"""Organize Satori-preprocessed fNIRS files into the BIDS derivatives structure.

Works for both the SD and the ID dataset: select the matching preprocessed
folder (SD_PPT or ID_PPT) and the matching BIDS root directory.

Expected input: a flat folder with files named like
sub-001_task-motor_run-1_nirs_PPT.snirf
"""

import json
import os
import re
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import pandas as pd

# =================================================================
# METADATA
# =================================================================

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

derivative_dataset_info = {
    "Name": "PREPROCESSED: fNIRS sensorimotor tasks: Hand clenching & Finger tapping",
    "BIDSVersion": "1.11.1",
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": "Satori",
            "Version": "2.2.4",
            "Description": "Preprocessing pipeline: 1. Raw to Concentration Changes, 2. Spike Removal, 3. TDDR, 4. Short-channel regressioN, 5. Bandpass Filter (0.01-0.09 Hz) + Detrending, 6. Z-Transformation, 7. Trimming (10s pre first, 20s post last event).",
        }
    ],
    "SourceDatasets": [
        {"Description": "Raw fNIRS data located in the root BIDS directory"}
    ],
}

PIPELINE_NAME = "nirs-preproc"

FILENAME_PATTERN = re.compile(
    r"sub-(?P<sub>\d+)_task-motor_run-(?P<run>\d+)_nirs_PPT\.snirf$",
    re.IGNORECASE,
)


def get_sidecar_metadata(bids_sub, run):
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


def read_exclude_ids(excel_path):
    """Return the digits of all participant IDs marked in the 'exclude' column."""
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["participant_id", "id", "subject"]), df.columns[0])

    if "exclude" not in df.columns:
        return []
    raw = df[df["exclude"].notna()][id_col].astype(str).tolist()
    return [re.sub(r"\D", "", x) for x in raw]


def organize_fnirs_derivatives():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo("Step 1", "Select the folder with the preprocessed Satori files (SD_PPT or ID_PPT).")
    source_root = filedialog.askdirectory(title="Preprocessed files folder")
    if not source_root:
        return

    messagebox.showinfo("Step 2", "Select the BIDS root directory (SD or ID) that matches the source folder.")
    base_dest = filedialog.askdirectory(title="BIDS root directory")
    if not base_dest:
        return

    messagebox.showinfo("Step 3", "Select the Excel file with the 'exclude' column.")
    exclude_path = filedialog.askopenfilename(
        title="Exclusion list", filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not exclude_path:
        return

    exclude_ids = read_exclude_ids(exclude_path)

    dest_root = Path(base_dest) / "derivatives" / PIPELINE_NAME
    dest_root.mkdir(parents=True, exist_ok=True)

    with open(dest_root / "README", "w", encoding="utf-8") as f:
        f.write("# Preprocessing Description\n\n")
        f.write(PREPROCESSING_DETAILS)

    with open(dest_root / "dataset_description.json", "w", encoding="utf-8") as f:
        json.dump(derivative_dataset_info, f, indent=4)

    processed_count = 0
    skipped_count = 0
    unmatched_files = []

    for filename in sorted(os.listdir(source_root)):
        src_file_path = os.path.join(source_root, filename)
        if not os.path.isfile(src_file_path):
            continue

        match = FILENAME_PATTERN.match(filename)
        if not match:
            unmatched_files.append(filename)
            continue

        sub_digits = match.group("sub")
        run_number = match.group("run")

        if sub_digits in exclude_ids:
            skipped_count += 1
            continue

        bids_sub = f"sub-{sub_digits}"
        target_dir = dest_root / bids_sub / "nirs"
        target_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{bids_sub}_task-motor_run-{run_number}_desc-preproc_nirs"
        shutil.copy2(src_file_path, target_dir / f"{base_name}.snirf")

        with open(target_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(get_sidecar_metadata(bids_sub, run_number), f, indent=4)

        processed_count += 1

    if unmatched_files:
        print("Files with an unexpected name format (not copied):")
        for name in unmatched_files:
            print(f"  {name}")

    messagebox.showinfo(
        "Done",
        f"Export finished!\n"
        f"Processed files: {processed_count}\n"
        f"Excluded (exclude list): {skipped_count}\n"
        f"Unrecognized file names: {len(unmatched_files)}",
    )


if __name__ == "__main__":
    organize_fnirs_derivatives()
