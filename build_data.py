"""Precompute static data bundles for the fNIRS dashboard.

Reads every participant's SD fNIRS data from data/Pxxx/Pxxx_SD_X.npy and
writes:
    public/data/meta.json          — subjects, channels, conditions, shape
    public/data/grand_avg.json     — grand avg ± between-subject SE per condition
    public/data/subject_avg.bin    — (n_subjects, 2, 26, 306) float32
    public/data/subjects/Pxxx.bin  — (20, 26, 306) float32 single-trial data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "public" / "data"
SUBJECTS_DIR = OUT_DIR / "subjects"

N_CHANNELS = 26
N_TIMES = 306
N_TRIALS = 20
CONDITIONS = {"1": "Hand Clenching", "2": "Finger Tapping"}


def channel_info() -> list[dict]:
    info = []
    for i in range(N_CHANNELS):
        chromo = "HbO" if i < N_CHANNELS // 2 else "HbR"
        pair = (i % (N_CHANNELS // 2)) + 1
        info.append({"index": i, "name": f"{chromo}{pair:02d}", "chromophore": chromo, "pair": pair})
    return info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("P"))
    print(f"Found {len(subjects)} subjects")

    subj_avg = np.zeros((len(subjects), 2, N_CHANNELS, N_TIMES), dtype=np.float32)

    for si, subj in enumerate(subjects):
        X = np.load(DATA_DIR / subj / f"{subj}_SD_X.npy").astype(np.float32)
        y = np.load(DATA_DIR / subj / f"{subj}_SD_y.npy", allow_pickle=True)

        if X.shape != (N_TRIALS, N_CHANNELS, N_TIMES):
            raise ValueError(f"{subj}: unexpected X shape {X.shape}")
        if y.shape != (N_TRIALS,):
            raise ValueError(f"{subj}: unexpected y shape {y.shape}")

        X.tofile(SUBJECTS_DIR / f"{subj}.bin")

        mask1 = y == "1"
        mask2 = y == "2"
        subj_avg[si, 0] = X[mask1].mean(axis=0)
        subj_avg[si, 1] = X[mask2].mean(axis=0)

        (SUBJECTS_DIR / f"{subj}_labels.json").write_text(json.dumps(y.tolist()))

        if (si + 1) % 20 == 0 or si == len(subjects) - 1:
            print(f"  processed {si + 1}/{len(subjects)}")

    subj_avg.tofile(OUT_DIR / "subject_avg.bin")

    grand_mean = subj_avg.mean(axis=0)
    grand_se = subj_avg.std(axis=0, ddof=1) / np.sqrt(len(subjects))

    grand = {
        "shape": [2, N_CHANNELS, N_TIMES],
        "mean": grand_mean.astype(np.float32).round(5).tolist(),
        "se": grand_se.astype(np.float32).round(5).tolist(),
    }
    (OUT_DIR / "grand_avg.json").write_text(json.dumps(grand))

    meta = {
        "subjects": subjects,
        "n_subjects": len(subjects),
        "n_channels": N_CHANNELS,
        "n_times": N_TIMES,
        "n_trials": N_TRIALS,
        "channels": channel_info(),
        "conditions": [
            {"code": "1", "name": CONDITIONS["1"]},
            {"code": "2", "name": CONDITIONS["2"]},
        ],
        "subject_avg_shape": [len(subjects), 2, N_CHANNELS, N_TIMES],
        "subject_avg_dtype": "float32",
        "single_trial_shape": [N_TRIALS, N_CHANNELS, N_TIMES],
        "single_trial_dtype": "float32",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    total = (
        sum(f.stat().st_size for f in SUBJECTS_DIR.glob("*.bin"))
        + (OUT_DIR / "subject_avg.bin").stat().st_size
        + (OUT_DIR / "grand_avg.json").stat().st_size
        + (OUT_DIR / "meta.json").stat().st_size
    )
    print(f"\nWrote {OUT_DIR} ({total / 1e6:.1f} MB total)")


if __name__ == "__main__":
    main()
