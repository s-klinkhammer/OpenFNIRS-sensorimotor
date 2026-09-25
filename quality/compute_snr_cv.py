#!/usr/bin/env python
"""
compute_snr_cv.py
==================
Computes per-channel signal quality metrics (SNR, CV, cardiac-band SNR, SCI)
for the data-quality report, combining:

  - trimmed RAW intensity .snirf files   (produced by quality/Trim.flow)
  - trimmed OD SCI .txt files            (produced by quality/Trim_OD.flow
                                           + quality/SCI.flow)

See quality_readme.md in this directory for the full Satori workflow that
produces these inputs.

Outputs (written to --out-dir, default results/quality/):
    fNIRSQA_output.csv          — channel-level SNR/CV, one row per channel per file
    all_runs_coupling_metrics.csv — source-detector pair level SCI/cSNR + quality flags

Usage
-----
    python quality/compute_snr_cv.py
    python quality/compute_snr_cv.py --snirf-dir /custom/path --sci-dir /custom/path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snirf_io import load_snirf

# This script lives in quality/, so the repo root is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Flexible BIDS Directory Detection (same pattern as build_data.py,
# classification/generate_npy.py, glm_analysis/generate_msd.py) ---
DATA_BASE = ROOT_DIR / "data"
if (DATA_BASE / "bids_fnirs_sensorimotor_dataset").exists():
    BIDS_DIR = DATA_BASE / "bids_fnirs_sensorimotor_dataset"
else:
    BIDS_DIR = DATA_BASE

DEFAULT_SNIRF_DIR = BIDS_DIR / "derivatives" / "quality" / "trim_raw"
DEFAULT_SCI_DIR = BIDS_DIR / "derivatives" / "quality" / "trim_od"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "quality"

# Parameters per Raible et al.
CARDIAC_BAND = (0.80, 2.00)
NOISE_BANDS = [(0.10, 0.50), (2.10, 2.40)]
WELCH_NPERSEG = 300
DEFAULT_FS = 12.6


def parse_sci_file(sci_txt_path: Path) -> dict:
    """Read per-channel SCI values from the .txt file exported by Satori's
    SCI Channel Rejection node. Returns {channel_index: sci_value}."""
    sci_dict = {}
    if not sci_txt_path.exists():
        return sci_dict

    with open(sci_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if "CH:" in line and "SCI:" in line:
                parts = line.split()
                # e.g. ['CH:', '1', 'SCI:', '0.720734']
                ch_idx = int(parts[parts.index("CH:") + 1])
                sci_val = float(parts[parts.index("SCI:") + 1])
                sci_dict[ch_idx] = sci_val
    return sci_dict


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
    """Trapezoidal integration of the PSD within a frequency band."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return 1e-12
    return float(np.trapezoid(psd[mask], freqs[mask]))


def classify_quality(sci: float, csnr_db: float):
    """Classify SCI, SNR, and overall quality against fixed thresholds."""
    sci_q = "strong" if sci >= 0.75 else ("usable" if sci >= 0.50 else "poor")
    snr_q = "good" if csnr_db >= 13.0 else ("usable" if csnr_db >= 2.5 else "poor")

    if sci_q == "strong" and snr_q == "good":
        overall = "good"
        suggestion = "keep"
    elif sci_q in ["strong", "usable"] and snr_q in ["good", "usable"]:
        overall = "usable"
        suggestion = "keep"
    else:
        overall = "poor"
        suggestion = "review_or_reject"

    return sci_q, snr_q, overall, suggestion


def process_snirf_file(snirf_path: Path, sci_dir: Path, default_fs: float = DEFAULT_FS):
    """Process one trimmed-RAW SNIRF file, combined with its matching SCI .txt
    (produced from the corresponding trimmed-OD file) from sci_dir."""
    data = load_snirf(str(snirf_path))
    file_name = snirf_path.name  # e.g. sub-001_..._TRIM.snirf

    # Match the SCI file exported alongside the corresponding trimmed-OD file.
    base_stem = snirf_path.stem
    sci_file = sci_dir / f"{base_stem}_OD_rejectedChannels_SCI.txt"
    sci_lookup = parse_sci_file(sci_file)
    if not sci_lookup:
        print(f"  WARN {file_name}: no matching SCI file found at {sci_file.name} in {sci_dir}")

    intensity = data.intensity
    wavelength_nm = np.round(data.wavelength_nm).astype(int)
    n_samples, n_channels = intensity.shape

    fs = getattr(data, "fs", None) or getattr(data, "sampling_rate", None) or default_fs
    duration_s = float(n_samples / fs)

    source_index = getattr(data, "source_index", np.zeros(n_channels, dtype=int))
    detector_index = getattr(data, "detector_index", np.zeros(n_channels, dtype=int))

    # --- 1. Channel-level SNR/CV ---
    qa_rows = []
    for ch in range(n_channels):
        sig = intensity[:, ch]

        n_nan = int(np.isnan(sig).sum())
        if n_nan > 0:
            print(f"  WARN {file_name} CH{ch + 1}: {n_nan} NaN values in signal, ignoring")

        m_val = np.nanmean(sig)
        s_val = np.nanstd(sig)

        if s_val > 0 and m_val > 0:
            snr_val = 20 * np.log10(m_val / s_val)
        else:
            snr_val = np.nan
            print(f"  WARN {file_name} CH{ch + 1}: mean={m_val:.4g}, std={s_val:.4g} "
                  f"-> SNR undefined (mean<=0 or std<=0), setting NaN")

        cv_val = (s_val / m_val) * 100 if m_val > 0 else np.nan

        src = int(source_index[ch])
        det = int(detector_index[ch])

        qa_rows.append({
            "Filename": file_name,
            "channel_index": ch + 1,
            "source_index": src,
            "detector_index": det,
            "channel_label": f"S{src}-D{det}",
            "wavelength": float(wavelength_nm[ch]),
            "SNR": snr_val,
            "CV": cv_val
        })
    df_qa = pd.DataFrame(qa_rows)

    # --- 2. Source-detector pair level: SCI + cardiac-band SNR ---
    pair_keys = list(zip(source_index, detector_index))
    unique_pairs = sorted(set(pair_keys))

    metrics_rows = []
    for pair_idx, (src, det) in enumerate(unique_pairs, start=1):
        ch_indices = np.where([(s == src and d == det) for s, d in pair_keys])[0]
        if len(ch_indices) != 2:
            continue

        ch_wl1, ch_wl2 = ch_indices[0], ch_indices[1]

        # SCI values are 1-indexed per channel pair in the Satori export.
        sci_val = sci_lookup.get(pair_idx, 0.0)

        cardiac_powers, noise_powers = [], []
        for ch in [ch_wl1, ch_wl2]:
            s_demean = intensity[:, ch].astype(float) - intensity[:, ch].mean()
            freqs, psd = welch(s_demean, fs=fs, nperseg=min(WELCH_NPERSEG, len(s_demean)))

            cardiac_powers.append(_band_power(freqs, psd, CARDIAC_BAND))
            noise_powers.append(sum(_band_power(freqs, psd, b) for b in NOISE_BANDS))

        c_mean = np.mean(cardiac_powers)
        n_mean = np.mean(noise_powers)
        csnr_db = 10 * np.log10(c_mean / n_mean) if n_mean > 0 else 0

        sci_q, snr_q, overall_q, suggestion = classify_quality(sci_val, csnr_db)

        data_key = getattr(data, "data_key", "data1")

        metrics_rows.append({
            "file_name": file_name,
            "data_key": data_key,
            "sampling_rate_hz": fs,
            "n_samples": n_samples,
            "duration_s": duration_s,
            "source_index": int(src),
            "detector_index": int(det),
            "channel_label": f"S{int(src)}-D{int(det)}",
            "wl1_nm": float(wavelength_nm[ch_wl1]),
            "wl2_nm": float(wavelength_nm[ch_wl2]),
            "SCI": sci_val,
            "couplingSNR_dB": csnr_db,
            "heart_low_hz": CARDIAC_BAND[0],
            "heart_high_hz": CARDIAC_BAND[1],
            "noise_bands": "0.10-0.50; 2.10-2.40",
            "nperseg": WELCH_NPERSEG,
            "bandpass_order": 3,
            "sci_quality": sci_q,
            "snr_quality": snr_q,
            "overall_quality": overall_q,
            "keep_suggestion": suggestion
        })

    df_metrics = pd.DataFrame(metrics_rows)
    return df_qa, df_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snirf-dir", type=Path, default=DEFAULT_SNIRF_DIR,
                         help=f"Folder with trimmed-RAW .snirf files (default: {DEFAULT_SNIRF_DIR.relative_to(ROOT_DIR)})")
    parser.add_argument("--sci-dir", type=Path, default=DEFAULT_SCI_DIR,
                         help=f"Folder with *_OD_rejectedChannels_SCI.txt files (default: {DEFAULT_SCI_DIR.relative_to(ROOT_DIR)})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help=f"Where to write the output CSVs (default: {DEFAULT_OUT_DIR.relative_to(ROOT_DIR)})")
    args = parser.parse_args()

    if not args.snirf_dir.exists():
        print(f"Error: SNIRF directory not found at '{args.snirf_dir}'")
        print("Run quality/Trim.flow in Satori first — see quality/quality_readme.md.")
        sys.exit(1)
    if not args.sci_dir.exists():
        print(f"Error: SCI directory not found at '{args.sci_dir}'")
        print("Run quality/Trim_OD.flow + quality/SCI.flow in Satori first — see quality/quality_readme.md.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_qa, all_metrics = [], []
    snirf_files = sorted(args.snirf_dir.glob("*.snirf"))

    print(f"Found {len(snirf_files)} SNIRF files in {args.snirf_dir}. Processing...\n")

    for f in snirf_files:
        try:
            print(f"Processing: {f.name}...")
            df_qa, df_metrics = process_snirf_file(f, args.sci_dir)
            all_qa.append(df_qa)
            all_metrics.append(df_metrics)
        except Exception as e:
            print(f"  --> ERROR on {f.name}: {e}")

    if not all_qa:
        print("\nNo files processed — nothing written.")
        sys.exit(1)

    df_all_qa = pd.concat(all_qa, ignore_index=True)
    df_all_metrics = pd.concat(all_metrics, ignore_index=True)

    df_all_qa.to_csv(args.out_dir / "fNIRSQA_output.csv", index=False)
    df_all_metrics.to_csv(args.out_dir / "all_runs_coupling_metrics.csv", index=False)

    print(f"\nDone. Wrote both CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
