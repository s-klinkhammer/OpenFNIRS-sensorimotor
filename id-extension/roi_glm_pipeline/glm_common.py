"""
Shared configuration and data-loading helpers for the ROI GLM analysis scripts.

All scripts in this repository read the same input: one GLM result workbook
per participant and run (BIDS-style file names, see ``find_result_file``).
Everything that is shared between the scripts (study constants, command-line
options, file discovery, reading of T- and p-values, run-combination helpers)
lives here so that it is defined exactly once.

Expected input workbook layout (one workbook per participant and run)
---------------------------------------------------------------------
File name:  ``sub-<NNN>_..._run-<R>_..._GLM_Results.xlsx``
            e.g. ``sub-001_task-motor_run-1_nirs_PPT_GLM_Results.xlsx``
Sheets:     "Oxy GLM Contrast Results" (HbO) and
            "DeOxy GLM Contrast Results" (HbR)
Rows:       one row per channel ("CH 1", "CH 2", ...), starting at
            ``DATA_START_ROW``; the first column holds the channel label.
Columns:    T-value / p-value per contrast, see ``CONTRAST_COLUMNS``.
"""

import argparse
import os
from pathlib import Path

import openpyxl

from alpha_correction_utils import combine_two_run_pvalue_fisher, combine_two_run_tvalue

# =============================================================================
# Study constants (adapt to your data set)
# =============================================================================
DEFAULT_N_PARTICIPANTS = 109
ALPHA = 0.05

RUNS = ["RunA", "RunB"]
# Mapping of the two runs to the BIDS run entity used in the file names.
RUN_TO_BIDS = {"RunA": "run-1", "RunB": "run-2"}

# Chromophore key -> name of the sheet in the GLM result workbooks.
CHROMOPHORES = {
    "Oxy": "Oxy GLM Contrast Results",
    "DeOxy": "DeOxy GLM Contrast Results",
}
CHROMOPHORE_PAPER_LABEL = {"Oxy": "HbO", "DeOxy": "HbR"}

# Channels selected based on M1/S1 coverage.
#   SD & ID channels: available in both the standard-density (SD) and the
#                     increased-density (ID) montage.
#   ID-only channels: only available in the increased-density montage.
SD_AND_ID_CHANNELS = [2, 6, 16]
ID_ONLY_CHANNELS = [4, 10, 13]
CHANNELS_OF_INTEREST = ID_ONLY_CHANNELS + SD_AND_ID_CHANNELS

CONTRASTS = {
    "Contrast1": "Contrast 1 (Hand clenching > Baseline)",
    "Contrast2": "Contrast 2 (Finger tapping > Baseline)",
    "Contrast3": "Contrast 3 (Hand clenching > Finger tapping)",
}
# Only the two "> Baseline" contrasts are used for the significance analyses.
CONTRASTS_VS_BASELINE = ["Contrast1", "Contrast2"]
CONTRAST_TASK_LABEL = {
    "Contrast1": "Hand Clenching",
    "Contrast2": "Finger Tapping",
}

# 1-indexed (T-value column, p-value column) per contrast in the input sheets.
CONTRAST_COLUMNS = {
    "Contrast1": (3, 4),
    "Contrast2": (7, 8),
    "Contrast3": (11, 12),
}
DATA_START_ROW = 3     # row of "CH 1" in the input sheets
MAX_CHANNEL_ROWS = 20  # number of channel rows read per sheet


# =============================================================================
# Command-line handling
# =============================================================================
def add_common_args(parser):
    """Add the input/output/participant options shared by all scripts."""
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(os.environ.get("GLM_INPUT_DIR", "data/ID_SingleSubjectGLM")),
        help="Folder with the per-participant GLM result workbooks "
             "(default: $GLM_INPUT_DIR or ./data/ID_SingleSubjectGLM).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("GLM_OUTPUT_DIR", "output")),
        help="Folder for the result files; created if missing "
             "(default: $GLM_OUTPUT_DIR or ./output).",
    )
    parser.add_argument(
        "--n-participants", type=int, default=DEFAULT_N_PARTICIPANTS,
        help=f"Number of participants, numbered 1..N (default: {DEFAULT_N_PARTICIPANTS}).",
    )
    return parser


def participant_ids(n_participants):
    """['P001', 'P002', ...] - the participant labels used in all outputs."""
    return [f"P{i:03d}" for i in range(1, n_participants + 1)]


# =============================================================================
# Reading the GLM result workbooks
# =============================================================================
def extract_channel_values(ws, channels):
    """Return ``{channel_num: {contrast_key: {"t": T, "p": p}}}`` for the
    requested channels of one result sheet."""
    result = {}
    for row in ws.iter_rows(min_row=DATA_START_ROW,
                            max_row=DATA_START_ROW + MAX_CHANNEL_ROWS - 1):
        ch_label = row[0].value
        if not ch_label:
            continue
        try:
            ch_num = int(str(ch_label).replace("CH", "").strip())
        except ValueError:
            continue
        if ch_num not in channels:
            continue
        result[ch_num] = {
            contrast_key: {"t": row[t_col - 1].value, "p": row[p_col - 1].value}
            for contrast_key, (t_col, p_col) in CONTRAST_COLUMNS.items()
        }
    return result


def find_result_file(all_files, participant_number, run):
    """Locate the result workbook of one participant/run (BIDS naming).
    Returns a Path, or None if no file matches."""
    sub = f"sub-{participant_number:03d}"
    run_tag = RUN_TO_BIDS[run]
    matches = [
        f for f in all_files
        if f.name.startswith(f"{sub}_")
        and f"_{run_tag}_" in f.name
        and f.name.endswith("GLM_Results.xlsx")
    ]
    if len(matches) > 1:
        print(f"WARNING: several files for {sub} {run_tag}, using the first: "
              f"{[f.name for f in matches]}")
    return matches[0] if matches else None


def load_glm_results(input_dir, n_participants):
    """
    Read T- and p-values of all channels of interest.

    Returns ``(p, t, missing_files, missing_sheets)`` where
        p[contrast][run][chromophore][pid][channel] = p-value (or None)
        t[contrast][run][chromophore][pid][channel] = T-value (or None)
    """
    p = {c: {r: {ch: {} for ch in CHROMOPHORES} for r in RUNS} for c in CONTRASTS}
    t = {c: {r: {ch: {} for ch in CHROMOPHORES} for r in RUNS} for c in CONTRASTS}
    missing_files, missing_sheets = [], []

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    # Ignore Excel lock files ("~$...").
    all_files = [f for f in input_dir.glob("*.xlsx") if not f.name.startswith("~$")]

    for i, pid in enumerate(participant_ids(n_participants), start=1):
        for run in RUNS:
            fpath = find_result_file(all_files, i, run)
            if fpath is None:
                missing_files.append(f"{pid}_{run} (sub-{i:03d}, {RUN_TO_BIDS[run]})")
                continue
            wb = openpyxl.load_workbook(fpath, data_only=True)
            for chromo, sheet_name in CHROMOPHORES.items():
                if sheet_name not in wb.sheetnames:
                    missing_sheets.append(f"{fpath.name} -> '{sheet_name}'")
                    continue
                channel_vals = extract_channel_values(wb[sheet_name], CHANNELS_OF_INTEREST)
                for ch_num, contrast_vals in channel_vals.items():
                    for contrast_key, tp in contrast_vals.items():
                        p[contrast_key][run][chromo].setdefault(pid, {})[ch_num] = tp["p"]
                        t[contrast_key][run][chromo].setdefault(pid, {})[ch_num] = tp["t"]
            wb.close()

    return p, t, missing_files, missing_sheets


def report_missing(missing_files, missing_sheets):
    """Print a short summary of missing input files / sheets."""
    if missing_files:
        print(f"\n{len(missing_files)} expected file(s) not found:")
        for m in missing_files:
            print(f"  - {m}")
    if missing_sheets:
        print(f"\n{len(missing_sheets)} missing sheet(s) in existing files:")
        for m in missing_sheets:
            print(f"  - {m}")


# =============================================================================
# Run-combination helpers shared by the analysis scripts
# =============================================================================
def combined_pvalues_per_channel(p_runa, p_runb, t_runa, t_runb, channel, pids,
                                 expected_sign=None):
    """
    Fisher-combined, direction-checked p-value of one channel for all
    participants (see ``combine_two_run_pvalue_fisher``).

    Returns ``({pid: combined_p}, {pid: status})``. Participants with missing
    data are absent from the first dict; ``status`` distinguishes
    "combined", "conflicting_direction", "wrong_direction" and "missing".
    """
    combined, status_by_pid = {}, {}
    for pid in pids:
        comb, status = combine_two_run_pvalue_fisher(
            p_runa.get(pid, {}).get(channel), p_runb.get(pid, {}).get(channel),
            t_runa.get(pid, {}).get(channel), t_runb.get(pid, {}).get(channel),
            expected_sign=expected_sign,
        )
        status_by_pid[pid] = status
        if comb is not None:
            combined[pid] = comb
    return combined, status_by_pid


def combined_tvalues_multi_chromo(t_by_chromo_runa, t_by_chromo_runb, channels, pids):
    """``{pid: {(channel, chromophore): mean(T_RunA, T_RunB)}}`` over BOTH
    chromophores; used to estimate M_eff of the HbO+HbR union family."""
    out = {}
    for pid in pids:
        d = {}
        for chromo in CHROMOPHORES:
            for ch in channels:
                d[(ch, chromo)] = combine_two_run_tvalue(
                    t_by_chromo_runa[chromo].get(pid, {}).get(ch),
                    t_by_chromo_runb[chromo].get(pid, {}).get(ch),
                )
        out[pid] = d
    return out


def build_parser(description):
    """ArgumentParser with the common options already added."""
    return add_common_args(argparse.ArgumentParser(description=description))
