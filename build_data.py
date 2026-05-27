"""Precompute the static dashboard bundle from the preprocessed BIDS dataset.

Reads:
    data/derivatives/nirs-preproc/sub-XXX/nirs/sub-XXX_task-motor_run-{1,2}_desc-preproc_nirs.snirf
    data/sub-XXX/nirs/sub-XXX_task-motor_run-{1,2}_events.tsv

Writes (under public/data/):
    meta.json
    grand_avg.json            — block average across subjects + between-subject SE
    subject_avg.bin           — (n_subjects, 2, 26, N_TIMES) float32
    subjects/sub-XXX.bin      — (n_trials, 26, N_TIMES) float32
    subjects/sub-XXX_labels.json  — list of trial_type strings

Condition labels come from the BIDS events.tsv `trial_type` column
(`hand_clenching` / `finger_tapping`). SNIRF stim group names are unreliable —
we pair stim onsets to events.tsv rows by sorted time order, which is robust
to per-subject counterbalancing of the starting condition.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BIDS = ROOT / "data"
DERIV = BIDS / "derivatives" / "nirs-preproc"
OUT = ROOT / "public" / "data"
SUBJECTS_OUT = OUT / "subjects"

# Epoch window: -5 s to +25 s around stim onset (matches the Satori pipeline)
TMIN, TMAX = -5.0, 25.0
N_PRE, N_POST = 51, 254   # samples; total = N_PRE + N_POST + 1 = 306 at ~10.17 Hz
N_TIMES = N_PRE + N_POST + 1

CONDITION_ORDER = ["hand_clenching", "finger_tapping"]   # cond axis [0]=Hand, [1]=Finger
COND_DISPLAY = {"hand_clenching": "Hand Clenching", "finger_tapping": "Finger Tapping"}


# Trigger value -> task (per manuscript):
#   1 = finger_tapping
#   2 = hand_clenching
#   3 = finger_tapping  (alternate recording code; observed in most subjects)
#   0 = false/noise trigger -> dropped
# The events.tsv `trial_type` STRINGS were generated with the wrong mapping
# and are systematically inverted; we ignore them and use the numeric `value`.
VALUE_TO_TASK = {1: "finger_tapping", 2: "hand_clenching", 3: "finger_tapping"}


def read_events_tsv(path: Path) -> list[tuple[float, str]]:
    rows = []
    with open(path) as f:
        header = f.readline().rstrip().split("\t")
        i_onset = header.index("onset")
        i_value = header.index("value")
        for line in f:
            cols = line.rstrip().split("\t")
            if len(cols) <= max(i_onset, i_value) or not cols[i_onset]:
                continue
            try:
                val = int(cols[i_value])
            except ValueError:
                continue
            task = VALUE_TO_TASK.get(val)
            if task is None:
                continue   # drops value=0 noise triggers
            rows.append((float(cols[i_onset]), task))
    rows.sort(key=lambda r: r[0])
    return rows


def load_snirf(path: Path):
    """Return (data float32 (n_time, 26) HbO-then-HbR, time, snirf_onsets sorted)."""
    with h5py.File(path, "r") as f:
        data = f["/nirs/data1/dataTimeSeries"][:].astype(np.float32)
        time = f["/nirs/data1/time"][:]
        ml_keys = sorted(
            (k for k in f["/nirs/data1"] if k.startswith("measurementList")),
            key=lambda k: int(k.replace("measurementList", "")),
        )
        chroma = [f[f"/nirs/data1/{k}/dataTypeLabel"][0].decode() for k in ml_keys]
        onsets = []
        for k in f["/nirs"]:
            if k.startswith("stim"):
                onsets.extend(f[f"/nirs/{k}/data"][:, 0].tolist())
        onsets.sort()
    hbo_idx = [i for i, c in enumerate(chroma) if c == "HbO"]
    hbr_idx = [i for i, c in enumerate(chroma) if c == "HbR"]
    if len(hbo_idx) != 13 or len(hbr_idx) != 13:
        raise ValueError(
            f"{path.name}: expected 13 HbO + 13 HbR channels, got {len(hbo_idx)} HbO + {len(hbr_idx)} HbR"
        )
    return data[:, hbo_idx + hbr_idx], time, onsets


def epoch_run(data, time, snirf_onsets, trial_types):
    """Build epochs by pairing snirf_onsets and trial_types in sorted time order."""
    n = min(len(snirf_onsets), len(trial_types))
    epochs, labels = [], []
    for k in range(n):
        onset = snirf_onsets[k]
        idx = int(np.argmin(np.abs(time - onset)))
        i0, i1 = idx - N_PRE, idx + N_POST + 1
        if i0 < 0 or i1 > data.shape[0]:
            continue
        epochs.append(data[i0:i1, :].T.astype(np.float32))    # (26, N_TIMES)
        labels.append(trial_types[k])
    if not epochs:
        return None, []
    return np.stack(epochs), labels


def process_subject(subj: str):
    deriv_dir = DERIV / subj / "nirs"
    raw_dir = BIDS / subj / "nirs"
    X_runs, y_runs = [], []
    for run in (1, 2):
        snirf = deriv_dir / f"{subj}_task-motor_run-{run}_desc-preproc_nirs.snirf"
        events_tsv = raw_dir / f"{subj}_task-motor_run-{run}_events.tsv"
        if not snirf.exists() or not events_tsv.exists():
            continue
        data, time, snirf_onsets = load_snirf(snirf)
        events = read_events_tsv(events_tsv)
        ep, labels = epoch_run(data, time, snirf_onsets, [e[1] for e in events])
        if ep is None:
            continue
        X_runs.append(ep)
        y_runs.extend(labels)
    if not X_runs:
        return None, []
    return np.concatenate(X_runs, axis=0), y_runs


def main():
    if SUBJECTS_OUT.exists():
        shutil.rmtree(SUBJECTS_OUT)   # clear old per-subject files
    SUBJECTS_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    subjects = sorted(d.name for d in DERIV.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    print(f"Found {len(subjects)} subjects under {DERIV.relative_to(ROOT)}")

    valid_subjects = []
    subj_avg_list = []
    n_trials_map = {}

    for si, subj in enumerate(subjects):
        X, y = process_subject(subj)
        if X is None or len(y) == 0:
            print(f"  skip {subj}: no usable data")
            continue
        unknown = {t for t in y if t not in CONDITION_ORDER}
        if unknown:
            print(f"  warn {subj}: unexpected trial_types {unknown}; keeping known only")
            keep = [i for i, t in enumerate(y) if t in CONDITION_ORDER]
            X = X[keep]
            y = [y[i] for i in keep]
        mask_hc = np.array([t == "hand_clenching" for t in y])
        mask_ft = np.array([t == "finger_tapping" for t in y])
        if mask_hc.sum() == 0 or mask_ft.sum() == 0:
            print(f"  skip {subj}: missing one condition")
            continue

        X.tofile(SUBJECTS_OUT / f"{subj}.bin")
        (SUBJECTS_OUT / f"{subj}_labels.json").write_text(json.dumps(y))

        avg = np.zeros((2, 26, N_TIMES), dtype=np.float32)
        avg[0] = X[mask_hc].mean(axis=0)
        avg[1] = X[mask_ft].mean(axis=0)
        subj_avg_list.append(avg)
        valid_subjects.append(subj)
        n_trials_map[subj] = len(y)

        if (si + 1) % 20 == 0 or si == len(subjects) - 1:
            print(f"  processed {si + 1}/{len(subjects)}")

    if not valid_subjects:
        raise RuntimeError("No subjects had usable data — nothing to write.")

    subj_avg = np.stack(subj_avg_list, axis=0)
    subj_avg.tofile(OUT / "subject_avg.bin")

    # Block average across subjects + between-subject SE
    block_mean = subj_avg.mean(axis=0)
    block_se = subj_avg.std(axis=0, ddof=1) / np.sqrt(len(valid_subjects))
    grand = {
        "shape": [2, 26, N_TIMES],
        "mean": block_mean.astype(np.float32).round(5).tolist(),
        "se": block_se.astype(np.float32).round(5).tolist(),
    }
    (OUT / "grand_avg.json").write_text(json.dumps(grand))

    channels = []
    for i in range(26):
        chromo = "HbO" if i < 13 else "HbR"
        pair = (i % 13) + 1
        channels.append({"index": i, "name": f"{chromo}{pair:02d}", "chromophore": chromo, "pair": pair})

    meta = {
        "subjects": valid_subjects,
        "n_subjects": len(valid_subjects),
        "n_channels": 26,
        "n_times": N_TIMES,
        "n_trials": 20,
        "n_trials_per_subject": n_trials_map,
        "channels": channels,
        "conditions": [
            {"code": "hand_clenching", "name": "Hand Clenching"},
            {"code": "finger_tapping", "name": "Finger Tapping"},
        ],
        "fs_nominal": 10.173,
        "tmin": TMIN,
        "tmax": TMAX,
        "subject_avg_shape": [len(valid_subjects), 2, 26, N_TIMES],
        "subject_avg_dtype": "float32",
        "single_trial_dtype": "float32",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    total = (
        sum(f.stat().st_size for f in SUBJECTS_OUT.glob("*"))
        + (OUT / "subject_avg.bin").stat().st_size
        + (OUT / "grand_avg.json").stat().st_size
        + (OUT / "meta.json").stat().st_size
    )
    print(f"\nWrote {OUT} ({total / 1e6:.1f} MB total, {len(valid_subjects)} subjects)")


if __name__ == "__main__":
    main()
