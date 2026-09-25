#!/usr/bin/env python
"""
generate_npy.py
===============
Precompute per-participant fNIRS epoched feature arrays (.npy) from BIDS 
derivatives for Riemannian cross-subject classification by directly loading
HDF5 / SNIRF data structures.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

# This script lives in classification/, so the repo root is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent
DERIVATIVES_PATH = ROOT_DIR / "data" / "derivatives" / "nirs-preproc"
OUTPUT_DIR = ROOT_DIR / "data"

CONDITION_LABEL = "SD"
TMIN, TMAX = -5.0, 25.0


# ============================================================================
# Robust SNIRF Loader
# ============================================================================

def load_snirf_data(snirf_path: str):
    """Read continuous data and stim onsets directly from SNIRF HDF5 layout."""
    with h5py.File(snirf_path, "r") as f:
        # Dynamisch nach 'nirs' oder 'nirs1' suchen
        nirs_key = next((k for k in ["nirs", "nirs1"] if k in f), None)
        if not nirs_key:
            # Fallback: Erster verfuegbarer Gruppenschluessel
            nirs_key = list(f.keys())[0] if f.keys() else None

        if not nirs_key:
            return None, None, None

        nirs = f[nirs_key]
        
        # Data1 oder erstes verfügbares Datenpaket finden
        data_key = next((k for k in ["data1", "data"] if k in nirs), None)
        if not data_key:
            data_keys = [k for k in nirs.keys() if k.startswith("data")]
            data_key = data_keys[0] if data_keys else None

        if not data_key:
            return None, None, None

        data = np.array(nirs[data_key]["dataTimeSeries"]).T  # (channels, time)
        time = np.array(nirs[data_key]["time"])

        # Events aus Stim-Gruppen laden
        onsets, labels = [], []
        stim_keys = [k for k in nirs.keys() if k.startswith("stim")]
        for idx, k in enumerate(stim_keys, start=1):
            st = nirs[k]
            if "data" in st:
                arr = np.array(st["data"])
                if arr.ndim == 2 and arr.shape[0] > 0:
                    onsets.extend(arr[:, 0])
                    labels.extend([idx] * len(arr))

    if not onsets:
        return None, None, None

    # Events nach Zeit sortieren
    sort_idx = np.argsort(onsets)
    onsets = np.array(onsets)[sort_idx]
    labels = np.array(labels)[sort_idx]

    return data, time, np.column_stack([onsets, labels])


def extract_epochs(data: np.ndarray, time: np.ndarray, events: np.ndarray):
    """Extract fixed-window epochs around stim onsets."""
    fs = 1.0 / np.mean(np.diff(time))
    n_pre = int(round(abs(TMIN) * fs))
    n_post = int(round(TMAX * fs))
    n_times = n_pre + n_post

    epochs, labels = [], []
    for onset, label in events:
        idx = np.searchsorted(time, onset)
        start, end = idx - n_pre, idx + n_post
        if start >= 0 and end <= data.shape[1]:
            ep = data[:, start:end]
            if ep.shape[1] == n_times:
                epochs.append(ep)
                labels.append(label)

    if not epochs:
        return None, None

    return np.array(epochs), np.array(labels)


# ============================================================================
# Core Preprocessing Pipeline
# ============================================================================

def process_participant(sub_id: str, derivatives_path: Path, output_dir: Path) -> bool:
    """Extract epochs for a single participant and save .npy feature arrays."""
    nirs_dir = derivatives_path / sub_id / "nirs"
    if not nirs_dir.exists():
        return False

    snirf_files = sorted(glob.glob(str(nirs_dir / "*.snirf")))
    if not snirf_files:
        return False

    Xs, ys = [], []
    for fname in snirf_files:
        try:
            data, time, events = load_snirf_data(fname)
            if data is None or events is None:
                continue

            epochs, labels = extract_epochs(data, time, events)
            if epochs is not None:
                Xs.append(epochs)
                ys.append(labels)
        except Exception as err:
            print(f"[{sub_id}] Error processing {Path(fname).name}: {err}")

    if not Xs:
        return False

    X = np.concatenate(Xs, axis=0).astype(np.float64)
    y = np.concatenate(ys).astype(int)

    sub_out_dir = output_dir / sub_id
    sub_out_dir.mkdir(parents=True, exist_ok=True)

    np.save(sub_out_dir / f"{sub_id}_{CONDITION_LABEL}_X.npy", X)
    np.save(sub_out_dir / f"{sub_id}_{CONDITION_LABEL}_y.npy", y)

    print(f"[{sub_id}] Successfully generated: X shape = {X.shape}, y shape = {y.shape}")
    return True


# ============================================================================
# Main Execution
# ============================================================================

def main():
    if not DERIVATIVES_PATH.exists():
        print(f"Error: BIDS derivatives directory not found at '{DERIVATIVES_PATH}'")
        print("Download the derivatives from Zenodo (see README) and unzip them")
        print(f"so that this path exists: {DERIVATIVES_PATH}")
        sys.exit(1)

    participants = sorted(
        d for d in os.listdir(DERIVATIVES_PATH)
        if d.startswith("sub-") and (DERIVATIVES_PATH / d).is_dir()
    )

    print("=" * 78)
    print(f"fNIRS Feature Extraction Pipeline ({len(participants)} participants found)")
    print(f"Input : {DERIVATIVES_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 78)

    processed_count = 0
    for sub in participants:
        if process_participant(sub, DERIVATIVES_PATH, OUTPUT_DIR):
            processed_count += 1

    print("\n" + "=" * 78)
    print(f"Completed! Processed {processed_count}/{len(participants)} participants.")
    print("You can now run: KMP_DUPLICATE_LIB_OK=TRUE python publish_riemann_ensemble.py")
    print("=" * 78)


if __name__ == "__main__":
    main()