#!/usr/bin/env python
"""
generate_msd.py
================
Builds a Satori Multi-Study GLM definition (.msd) file listing every
preprocessed run under data/derivatives/nirs-preproc/, so they don't have
to be added one by one via Satori's "Add Runs..." dialog.

Usage
-----
    python glm_analysis/generate_msd.py

Then in Satori: Multi-Study GLM -> Studies tab -> "Load .MSD..." and pick
the generated file. Conditions/contrasts still need to be configured in
Satori afterwards (not stored by this script).
"""
from __future__ import annotations

import sys
from pathlib import Path

# This script lives in glm_analysis/, so the repo root is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent
DERIVATIVES_PATH = ROOT_DIR / "data" / "derivatives" / "nirs-preproc"

# Written into data/ (already gitignored) since it embeds this machine's
# absolute paths and should never be committed.
OUTPUT_MSD = ROOT_DIR / "data" / "group_glm_runs.msd"

SEPARATE_PREDICTORS = 0  # matches Satori's "Separate subject predictors" setting


def main():
    if not DERIVATIVES_PATH.exists():
        print(f"Error: derivatives directory not found at '{DERIVATIVES_PATH}'")
        print("Download the derivatives from Zenodo first "
              "(see classification/classification_readme.md).")
        sys.exit(1)

    snirf_files = sorted(DERIVATIVES_PATH.glob("sub-*/nirs/*.snirf"))
    if not snirf_files:
        print(f"Error: no .snirf files found under '{DERIVATIVES_PATH}'")
        sys.exit(1)

    lines = [
        "FileVersion:\t1",
        f"SeparatePredictors:\t{SEPARATE_PREDICTORS}",
        "",
        f"NrOfStudies:\t{len(snirf_files)}",
    ]
    for f in snirf_files:
        # Satori expects forward slashes, even on Windows.
        posix_path = str(f.resolve()).replace("\\", "/")
        lines.append(f'"{posix_path}"')

    OUTPUT_MSD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(snirf_files)} runs to: {OUTPUT_MSD}")
    print('Open Satori -> Multi-Study GLM -> Studies tab -> "Load .MSD..."')
    print("to import them, then configure conditions/contrasts as usual.")


if __name__ == "__main__":
    main()
