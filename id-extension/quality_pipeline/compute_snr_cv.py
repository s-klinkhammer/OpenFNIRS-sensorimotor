"""Compute channel-level SNR and CV and pair-level coupling SNR (cSNR).

SNR and CV are computed per channel (one wavelength) from raw intensity.
cSNR is computed per source-detector pair from the cardiac band versus two
noise bands (Raible et al.). SCI values are read from the text files exported
by Satori (see README); they were calculated on motion-corrected data.

Outputs (written to --out-dir):
    fNIRSQA_output.csv              one row per channel and wavelength
    all_runs_coupling_metrics.csv   one row per source-detector pair

Usage:
    python compute_snr_cv.py --snirf-dir RAW_TRIM --sci-dir SCI --out-dir OUT
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from snirf_io import load_snirf

CARDIAC_BAND = (0.80, 2.00)                 # Hz
NOISE_BANDS = [(0.10, 0.50), (2.10, 2.40)]  # Hz
WELCH_NPERSEG = 300


def parse_sci_file(sci_txt_path: Path) -> dict:
    """Read SCI values from a Satori text file. Returns {pair_index: sci}.

    Expected line format: ``CH: 1 SCI: 0.720734``
    """
    sci = {}
    with open(sci_txt_path, "r") as f:
        for line in f:
            if "CH:" in line and "SCI:" in line:
                parts = line.split()
                sci[int(parts[parts.index("CH:") + 1])] = float(parts[parts.index("SCI:") + 1])
    return sci


def find_sci_file(snirf_path: Path, sci_dir: Path):
    """Find the SCI text file matching a SNIRF file by subject and run number."""
    m_sub = re.search(r"(sub-\d+)", snirf_path.name)
    m_run = re.search(r"run-(\d+)", snirf_path.name)
    if not (m_sub and m_run):
        return None
    sub, run = m_sub.group(1), m_run.group(1)

    candidates = sorted(
        f for f in sci_dir.glob("*.txt")
        if re.match(rf"^{sub}(?!\d)", f.name) and re.search(rf"run-{run}(?!\d)", f.name)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"  WARN: several SCI files for {snirf_path.name}, using {candidates[0].name}")
    return candidates[0]


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
    """Integrate the PSD (trapezoidal rule) within a frequency band."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return 1e-12
    return float(np.trapezoid(psd[mask], freqs[mask]))


def classify_quality(sci: float, csnr_db: float):
    """Classify SCI and cSNR and derive an overall quality label."""
    sci_q = "strong" if sci >= 0.75 else ("usable" if sci >= 0.50 else "poor")
    snr_q = "good" if csnr_db >= 13.0 else ("usable" if csnr_db >= 2.5 else "poor")

    if sci_q == "strong" and snr_q == "good":
        return sci_q, snr_q, "good", "keep"
    if sci_q in ("strong", "usable") and snr_q in ("good", "usable"):
        return sci_q, snr_q, "usable", "keep"
    return sci_q, snr_q, "poor", "review_or_reject"


def process_snirf_file(snirf_path: Path, sci_dir: Path):
    """Return (channel-level QA table, pair-level coupling table) for one file."""
    data = load_snirf(snirf_path)
    file_name = snirf_path.name

    sci_file = find_sci_file(snirf_path, sci_dir)
    if sci_file is None:
        raise FileNotFoundError(f"No SCI file for {file_name} in {sci_dir}")
    sci_lookup = parse_sci_file(sci_file)

    intensity = data.intensity
    wavelength_nm = np.round(data.wavelength_nm).astype(int)
    n_samples, n_channels = intensity.shape
    fs = data.fs
    duration_s = float(n_samples / fs)
    source_index = data.source_index
    detector_index = data.detector_index

    # Channel level: SNR (dB) = 20*log10(mean/std), CV (%) = std/mean*100
    qa_rows = []
    for ch in range(n_channels):
        sig = intensity[:, ch]

        n_nan = int(np.isnan(sig).sum())
        if n_nan > 0:
            print(f"  WARN {file_name} CH{ch + 1}: {n_nan} NaN values ignored")

        m_val = np.nanmean(sig)
        s_val = np.nanstd(sig)

        if s_val > 0 and m_val > 0:
            snr_val = 20 * np.log10(m_val / s_val)
        else:
            snr_val = np.nan
            print(f"  WARN {file_name} CH{ch + 1}: mean={m_val:.4g}, std={s_val:.4g}, SNR set to NaN")

        src, det = int(source_index[ch]), int(detector_index[ch])
        qa_rows.append({
            "Filename": file_name,
            "channel_index": ch + 1,
            "source_index": src,
            "detector_index": det,
            "channel_label": f"S{src}-D{det}",
            "wavelength": float(wavelength_nm[ch]),
            "SNR": snr_val,
            "CV": (s_val / m_val) * 100 if m_val > 0 else np.nan,
        })
    df_qa = pd.DataFrame(qa_rows)

    # Pair level: SCI from Satori file, cSNR from the two wavelengths.
    # SCI file entries are indexed by the position of the pair in the sorted
    # list of unique (source, detector) pairs.
    pair_keys = list(zip(source_index, detector_index))
    unique_pairs = sorted(set(pair_keys))

    metrics_rows = []
    for pair_idx, (src, det) in enumerate(unique_pairs, start=1):
        ch_indices = [i for i, key in enumerate(pair_keys) if key == (src, det)]
        if len(ch_indices) != 2:
            continue

        sci_val = sci_lookup.get(pair_idx, np.nan)
        if np.isnan(sci_val):
            print(f"  WARN {file_name} S{src}-D{det}: no SCI value (pair_idx={pair_idx})")

        cardiac_powers, noise_powers = [], []
        for ch in ch_indices:
            sig = intensity[:, ch].astype(float)
            freqs, psd = welch(sig - sig.mean(), fs=fs, nperseg=min(WELCH_NPERSEG, len(sig)))
            cardiac_powers.append(band_power(freqs, psd, CARDIAC_BAND))
            noise_powers.append(sum(band_power(freqs, psd, b) for b in NOISE_BANDS))

        c_mean, n_mean = np.mean(cardiac_powers), np.mean(noise_powers)
        csnr_db = 10 * np.log10(c_mean / n_mean) if n_mean > 0 else 0

        sci_q, snr_q, overall_q, suggestion = classify_quality(sci_val, csnr_db)

        metrics_rows.append({
            "file_name": file_name,
            "sampling_rate_hz": fs,
            "n_samples": n_samples,
            "duration_s": duration_s,
            "source_index": int(src),
            "detector_index": int(det),
            "channel_label": f"S{int(src)}-D{int(det)}",
            "wl1_nm": float(wavelength_nm[ch_indices[0]]),
            "wl2_nm": float(wavelength_nm[ch_indices[1]]),
            "SCI": sci_val,
            "couplingSNR_dB": csnr_db,
            "heart_low_hz": CARDIAC_BAND[0],
            "heart_high_hz": CARDIAC_BAND[1],
            "noise_bands": "; ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in NOISE_BANDS),
            "nperseg": WELCH_NPERSEG,
            "sci_quality": sci_q,
            "snr_quality": snr_q,
            "overall_quality": overall_q,
            "keep_suggestion": suggestion,
        })

    return df_qa, pd.DataFrame(metrics_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snirf-dir", type=Path, required=True,
                        help="Folder with raw (unprocessed) trimmed .snirf files")
    parser.add_argument("--sci-dir", type=Path, required=True,
                        help="Folder with Satori SCI .txt files")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output folder for the two CSV files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snirf_files = sorted(args.snirf_dir.glob("*.snirf"))
    print(f"{len(snirf_files)} SNIRF files found.\n")

    all_qa, all_metrics = [], []
    for f in snirf_files:
        try:
            print(f"Processing {f.name}")
            df_qa, df_metrics = process_snirf_file(f, args.sci_dir)
            all_qa.append(df_qa)
            all_metrics.append(df_metrics)
        except Exception as e:
            print(f"  ERROR in {f.name}: {e}")

    if not all_qa:
        raise SystemExit("No file was processed successfully.")

    pd.concat(all_qa, ignore_index=True).to_csv(args.out_dir / "fNIRSQA_output.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(args.out_dir / "all_runs_coupling_metrics.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
