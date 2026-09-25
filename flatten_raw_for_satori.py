#!/usr/bin/env python
"""
flatten_raw_for_satori.py
==========================
Copies every subject's raw .snirf file(s) out of the nested BIDS layout
(data/sub-XXX/nirs/*.snirf) into a single flat folder, so Satori's
"Load RAW fNIRS Dataset" node — whose file dialog is scoped to one folder
at a time (see preprocessing_readme.md / quality_readme.md) — can show and
select files across all subjects at once, instead of navigating into each
sub-XXX/nirs/ folder individually.

Filenames are already unique per subject+run (e.g.
sub-001_task-motor_run-1_nirs.snirf), so nothing is renamed or overwritten.

This does NOT copy the derivatives, events.tsv, or anything else — only the
raw .snirf files directly under sub-XXX/nirs/ (i.e. it does not touch
derivatives/nirs-preproc/, which lives in a different subtree and is never
matched by the sub-*/nirs/*.snirf glob used here).

IMPORTANT: this is a workaround for one specific UI limitation, not a
guarantee that Satori will batch-process a multi-file selection as separate
runs. Test it on a handful of subjects first (select e.g. 6-8 files for
3-4 subjects in the Load node and see what happens) before relying on it
for the full dataset -- if Satori doesn't treat a multi-selection as
independent per-file runs, you're back to loading one subject's pair at a
time, just without the folder navigation.

Usage
-----
    python flatten_raw_for_satori.py
    python flatten_raw_for_satori.py --dest data/raw_flat --clean
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# --- Flexible BIDS Directory Detection (same pattern as the other scripts) ---
DATA_BASE = ROOT_DIR / "data"
if (DATA_BASE / "bids_fnirs_sensorimotor_dataset").exists():
    BIDS_DIR = DATA_BASE / "bids_fnirs_sensorimotor_dataset"
else:
    BIDS_DIR = DATA_BASE

DEFAULT_DEST = ROOT_DIR / "data" / "raw_flat"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                         help=f"Folder to copy the flattened raw files into (default: {DEFAULT_DEST.relative_to(ROOT_DIR)})")
    parser.add_argument("--clean", action="store_true",
                         help="Delete the destination folder first if it already exists, "
                              "instead of copying into whatever's already there.")
    args = parser.parse_args()

    if not BIDS_DIR.exists():
        print(f"Error: BIDS directory not found at '{BIDS_DIR}'")
        print("Unzip the dataset into data/ first — see 'Getting the data' in the top-level README.md.")
        sys.exit(1)

    raw_files = sorted(BIDS_DIR.glob("sub-*/nirs/*.snirf"))
    if not raw_files:
        print(f"Error: no raw .snirf files found under '{BIDS_DIR}/sub-*/nirs/'.")
        print("(This only looks at raw sub-XXX folders directly under the BIDS root, not derivatives/.)")
        sys.exit(1)

    if args.clean and args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for f in raw_files:
        target = args.dest / f.name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(f, target)
        copied += 1

    print(f"Found {len(raw_files)} raw .snirf files under {BIDS_DIR.relative_to(ROOT_DIR)}/sub-*/nirs/")
    print(f"Copied {copied} file(s) to {args.dest.relative_to(ROOT_DIR)}"
          + (f" ({skipped} already present, skipped)" if skipped else ""))
    print("\nIn Satori's Load RAW fNIRS Dataset node, you can now browse to this one folder")
    print("and select across all subjects at once (test with a handful first — see the")
    print("docstring at the top of this script for why).")


if __name__ == "__main__":
    main()
