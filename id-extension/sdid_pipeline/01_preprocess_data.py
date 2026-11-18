"""
01_preprocess_data.py
Reads the preprocessed fNIRS data (BIDS derivatives, i.e. the *_desc-preproc_nirs.snirf
files from the data-preparation scripts), epochs them, and saves X.npy / y.npy per
participant and configuration (SD/ID) to config.PROCESSED_DIR.

Usage: python 01_preprocess_data.py [--tmin -5] [--tmax 25] [--chromophore-order {check,sort}]

Expected input
  separate layout: <BIDS_ROOT>/<SD|ID>/derivatives/nirs-preproc/sub-<label>/nirs/*_run-<..>_*_nirs.snirf
  single layout:   <BIDS_ROOT>/derivatives/nirs-preproc/sub-<label>/nirs/..._<entity>-<SD|ID>_..._nirs.snirf

Channel order (important for the block_covariances pipeline):
BlockCovariances(block_size=N) splits the channels into consecutive blocks of
size N. This only corresponds to "HbO | HbR" if X contains all HbO channels
first and all HbR channels after.
  --chromophore-order check (default): only verify the order and warn on deviation
  --chromophore-order sort           : stably sort the channels into HbO | HbR
The number of channels per chromophore is also checked against
config.BLOCK_SIZE_BY_CONDITION.

Trial order: runs are sorted by their run label so that trials appear in the
same order in SD and ID. 02_nested_cv.py uses the same random_state for both
conditions, so identical fold assignment requires identical trial order.
"""

import os
import re
import sys
import argparse
from glob import glob

import numpy as np
import mne

import config


def parse_args():
    p = argparse.ArgumentParser(description="BIDS fNIRS derivatives -> X.npy / y.npy")
    p.add_argument("--tmin", type=float, default=-5.0, help="epoch start in s (default -5)")
    p.add_argument("--tmax", type=float, default=25.0, help="epoch end in s (default 25)")
    p.add_argument("--chromophore-order", choices=["check", "sort"], default="check")
    return p.parse_args()


def subject_id(sub_label):
    """'sub-P001' / 'sub-001' / 'sub-1' -> 'P001' (the format expected by steps 02-04)."""
    label = sub_label.replace("sub-", "")
    digits = re.sub(r"\D", "", label)
    if label.upper().startswith("P") and digits:
        return f"P{digits.zfill(3)}"
    if label.isdigit():
        return f"P{label.zfill(3)}"
    return label


def run_label(filename):
    """Run entity from a BIDS file name (run-A, run-B, run-01, ...); used as sort key."""
    m = re.search(r"run-([A-Za-z0-9]+)", filename)
    if not m:
        raise ValueError(f"No 'run-<label>' found in file name: {filename}")
    return m.group(1).upper()


def find_snirf_files(cond):
    """Return {sub_label: [snirf paths]} for one configuration."""
    root = config.BIDS_ROOT
    if config.BIDS_LAYOUT == "separate":
        deriv = os.path.join(str(root), config.BIDS_COND_FOLDERS[cond], "derivatives", config.DERIVATIVES_PIPELINE)
        pattern = os.path.join(deriv, "sub-*", "**", "*_nirs.snirf")
        files = glob(pattern, recursive=True)
    elif config.BIDS_LAYOUT == "single":
        deriv = os.path.join(str(root), "derivatives", config.DERIVATIVES_PIPELINE)
        pattern = os.path.join(deriv, "sub-*", "**", "*_nirs.snirf")
        tag = f"{config.BIDS_COND_ENTITY}-{cond}"
        files = [f for f in glob(pattern, recursive=True) if tag in os.path.basename(f)]
    else:
        raise ValueError("config.BIDS_LAYOUT must be 'separate' or 'single'.")

    by_sub = {}
    for f in files:
        m = re.search(r"(sub-[A-Za-z0-9]+)", os.path.basename(f))
        if m:
            by_sub.setdefault(m.group(1), []).append(f)
    return by_sub


def check_or_sort_channels(raw, cond, mode, ident):
    """Check (or sort) the channel order HbO | HbR and the number of channels per chromophore."""
    types = raw.get_channel_types()
    n_hbo, n_hbr = types.count("hbo"), types.count("hbr")
    if n_hbo + n_hbr == 0:
        print(f"  [WARNING] {ident}: no hbo/hbr channels (types: {sorted(set(types))}) - "
              f"block check skipped.")
        return raw

    expected = config.BLOCK_SIZE_BY_CONDITION[cond]
    if n_hbo != expected or n_hbr != expected:
        print(f"  [WARNING] {ident}: {n_hbo} HbO / {n_hbr} HbR instead of {expected} channels each "
              f"(config.BLOCK_SIZE_BY_CONDITION['{cond}']).")

    blocks_ok = (set(types[:n_hbo]) == {"hbo"} and set(types[n_hbo:]) == {"hbr"})
    if mode == "sort" and not blocks_ok:
        order = [i for i, t in enumerate(types) if t == "hbo"] + [i for i, t in enumerate(types) if t == "hbr"]
        order += [i for i, t in enumerate(types) if t not in ("hbo", "hbr")]
        raw.reorder_channels([raw.ch_names[i] for i in order])
        print(f"  {ident}: channels sorted into HbO | HbR.")
    elif not blocks_ok:
        print(f"  [WARNING] {ident}: channels are NOT ordered as an HbO block followed by an HbR block - "
              f"the block_covariances pipeline would be invalid. Re-run with --chromophore-order sort.")
    return raw


def main():
    args = parse_args()
    config.require_bids_root()
    config.ensure_output_dirs()
    tmin, tmax = args.tmin, args.tmax
    output_root = str(config.PROCESSED_DIR)

    files_by_cond = {cond: find_snirf_files(cond) for cond in config.CONDITIONS}
    all_subs = sorted(set().union(*[set(d) for d in files_by_cond.values()]))
    if not all_subs:
        sys.exit(f"No *_nirs.snirf found under {config.BIDS_ROOT} (BIDS_LAYOUT='{config.BIDS_LAYOUT}').")
    print(f"{len(all_subs)} participants found.")

    for sub in all_subs:
        participant = subject_id(sub)
        run_order_by_cond = {}

        for cond in config.CONDITIONS:
            files = files_by_cond[cond].get(sub)
            if not files:
                print(f"No files found for {participant} {cond}, skipping.")
                continue
            files = sorted(files, key=lambda f: run_label(os.path.basename(f)))
            run_order_by_cond[cond] = [run_label(os.path.basename(f)) for f in files]

            Xs, ys = [], []
            for f in files:
                raw = mne.io.read_raw_snirf(f, optode_frame="unknown", preload=True, verbose=None)
                raw = check_or_sort_channels(raw, cond, args.chromophore_order,
                                             f"{participant} {cond} {os.path.basename(f)}")
                events, event_dict = mne.events_from_annotations(raw)
                epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                                   baseline=None, preload=True)
                data = epoch.get_data()
                print(f"{participant} {cond} - data shape: {data.shape}")

                id_to_name = {v: k for k, v in epoch.event_id.items()}
                y = np.array([id_to_name[l] for l in epoch.events[:, 2]])
                Xs.append(data)
                ys.append(y)

            X = np.concatenate(Xs, axis=0)
            y = np.concatenate(ys, axis=0)
            save_path = os.path.join(output_root, participant)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f"{participant}_{cond}_X.npy"), X)
            np.save(os.path.join(save_path, f"{participant}_{cond}_y.npy"), y)

        if len(run_order_by_cond) > 1:
            orders = list(run_order_by_cond.values())
            if any(o != orders[0] for o in orders):
                print(f"  [WARNING] {participant}: different run order between conditions "
                      f"({run_order_by_cond}) - fold assignment of SD/ID will NOT match exactly!")

    print("Epoching finished.")


if __name__ == "__main__":
    main()
