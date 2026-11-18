"""
01 - Per-contrast summary workbooks.

Builds one Excel workbook per contrast (``Summary_Contrast1.xlsx``, ...).
Each workbook contains:

``RunA_Oxy``, ``RunA_DeOxy``, ``RunB_Oxy``, ``RunB_DeOxy``
    Participants as rows, channels as column groups (T-value and p-value per
    channel). Next to the raw values, three live Excel formulas give the
    channel with the strongest activation, its T-value and its p-value.
    "Strongest activation" means the strongest T in the EXPECTED direction:
    the highest T for HbO, the most negative T for HbR (HbR decreases with
    activation).

``Best_Oxy``, ``Best_DeOxy``
    Per participant, the better of RunA/RunB (again in the expected
    direction) with its channel, T-value and p-value. These are live Excel
    formulas referencing the run sheets, so they recalculate if raw values are
    edited. Ranking channels is not a significance test; no correction applies.

``SigCount_Oxy``, ``SigCount_DeOxy``
    "% of participants with a significant effect per channel": Fisher's
    combined p-value of RunA+RunB, only if both runs agree in direction (and,
    for the "> Baseline" contrasts, match the expected direction), p < ALPHA.
    Descriptive and uncorrected (no multiple-comparison correction).

Usage:
    python 01_summary_per_contrast.py --input-dir <GLM results> --output-dir <out>
"""

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from alpha_correction_utils import expected_sign_for
from glm_common import (
    ALPHA, CHANNELS_OF_INTEREST, CHROMOPHORES, CONTRASTS, RUNS,
    build_parser, combined_pvalues_per_channel, load_glm_results,
    participant_ids, report_missing,
)

# Data rows in the summary sheets start at row 3 (rows 1-2 are headers).
SUMMARY_DATA_START_ROW = 3


def build_best_channel_formulas(row_idx, t_letters, p_letters, channels, expect_negative):
    """
    Excel formula strings (without leading '=') for one participant row:
    (channel_formula, best_t_formula, p_value_formula).

    ``t_letters`` / ``p_letters`` are the column letters of each channel's
    T-/p-value, in the same order as ``channels``. The placeholder
    ``BEST_T_CELL`` in the returned channel/p-value formulas must be replaced
    by the caller with the cell reference of the best-T formula.
    """
    t_refs = [f"{letter}{row_idx}" for letter in t_letters]
    agg = "MIN" if expect_negative else "MAX"   # HbR: most negative T is "best"
    best_t_formula = f"{agg}({','.join(t_refs)})"

    channel_expr = str(channels[-1])
    pvalue_expr = f"{p_letters[-1]}{row_idx}"
    for i in range(len(channels) - 2, -1, -1):
        channel_expr = f"IF({t_refs[i]}=BEST_T_CELL,{channels[i]},{channel_expr})"
        pvalue_expr = f"IF({t_refs[i]}=BEST_T_CELL,{p_letters[i]}{row_idx},{pvalue_expr})"
    return channel_expr, best_t_formula, pvalue_expr


def get_summary_column_layout(n_channels):
    """Column indices of (best channel, best T, corresponding p) in a run sheet."""
    channel_col = 2 + 2 * n_channels + 1
    return channel_col, channel_col + 1, channel_col + 2


def sheet_ref(sheet_name, cell):
    """Excel reference to ``cell`` on another sheet (quotes names with spaces/hyphens)."""
    if any(c in sheet_name for c in " -"):
        return f"'{sheet_name}'!{cell}"
    return f"{sheet_name}!{cell}"


def write_run_sheet(wb, sheet_name, t_map, p_map, pids, chromo_label):
    """One sheet with T-/p-values per participant and channel plus the
    best-channel formulas (RunA_<chromophore> / RunB_<chromophore>)."""
    expect_negative = (chromo_label == "DeOxy")
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)
    center = Alignment(horizontal="center")

    col = 2
    for ch in CHANNELS_OF_INTEREST:
        ws.cell(row=1, column=col, value=f"CH {ch}").font = bold
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + 1)
        ws.cell(row=1, column=col).alignment = center
        ws.cell(row=2, column=col, value="T-value").font = bold
        ws.cell(row=2, column=col + 1, value="p-value").font = bold
        col += 2

    n_channels = len(CHANNELS_OF_INTEREST)
    channel_col, best_t_col, pval_col = get_summary_column_layout(n_channels)
    ws.cell(row=2, column=channel_col,
            value="Channel with min T-value (HbR)" if expect_negative
            else "Channel with max T-value").font = bold
    ws.cell(row=2, column=best_t_col,
            value="Min T-value (HbR)" if expect_negative else "Max T-value").font = bold
    ws.cell(row=2, column=pval_col, value="Corresponding p-value").font = bold

    t_letters = [get_column_letter(2 + 2 * i) for i in range(n_channels)]
    p_letters = [get_column_letter(3 + 2 * i) for i in range(n_channels)]
    best_t_letter = get_column_letter(best_t_col)

    for offset, pid in enumerate(pids):
        row_idx = SUMMARY_DATA_START_ROW + offset
        ws.cell(row=row_idx, column=1, value=pid).font = bold
        col = 2
        for ch in CHANNELS_OF_INTEREST:
            ws.cell(row=row_idx, column=col, value=t_map.get(pid, {}).get(ch))
            ws.cell(row=row_idx, column=col + 1, value=p_map.get(pid, {}).get(ch))
            col += 2

        channel_expr, best_t_formula, pvalue_expr = build_best_channel_formulas(
            row_idx, t_letters, p_letters, CHANNELS_OF_INTEREST, expect_negative)
        best_t_cell = f"{best_t_letter}{row_idx}"
        ws.cell(row=row_idx, column=best_t_col, value=f"={best_t_formula}")
        ws.cell(row=row_idx, column=channel_col, value="=" + channel_expr.replace("BEST_T_CELL", best_t_cell))
        ws.cell(row=row_idx, column=pval_col, value="=" + pvalue_expr.replace("BEST_T_CELL", best_t_cell))

    for c in range(1, pval_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13


def write_best_of_both_runs_sheet(wb, sheet_name, run_a_sheet, run_b_sheet, pids, chromo_label):
    """Per participant: the better of RunA/RunB in the expected direction
    (HbO: larger T wins, HbR: more negative T wins), as live Excel formulas."""
    expect_negative = (chromo_label == "DeOxy")
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)

    channel_col, best_t_col, pval_col = get_summary_column_layout(len(CHANNELS_OF_INTEREST))
    channel_letter = get_column_letter(channel_col)
    best_t_letter = get_column_letter(best_t_col)
    pval_letter = get_column_letter(pval_col)

    for col, label in enumerate(
            ["Participant", "Best run", "Best channel", "Best T-value", "Corresponding p-value"], start=1):
        ws.cell(row=1, column=col, value=label).font = bold

    for offset, pid in enumerate(pids):
        src_row = SUMMARY_DATA_START_ROW + offset
        out_row = 2 + offset
        a_t = sheet_ref(run_a_sheet, f"{best_t_letter}{src_row}")
        b_t = sheet_ref(run_b_sheet, f"{best_t_letter}{src_row}")
        a_ch = sheet_ref(run_a_sheet, f"{channel_letter}{src_row}")
        b_ch = sheet_ref(run_b_sheet, f"{channel_letter}{src_row}")
        a_p = sheet_ref(run_a_sheet, f"{pval_letter}{src_row}")
        b_p = sheet_ref(run_b_sheet, f"{pval_letter}{src_row}")

        cmp_expr = f"{a_t}<={b_t}" if expect_negative else f"{a_t}>={b_t}"
        ws.cell(row=out_row, column=1, value=pid).font = bold
        ws.cell(row=out_row, column=2, value=f'=IF({cmp_expr},"RunA","RunB")')
        ws.cell(row=out_row, column=3, value=f"=IF({cmp_expr},{a_ch},{b_ch})")
        ws.cell(row=out_row, column=4, value=f"=IF({cmp_expr},{a_t},{b_t})")
        ws.cell(row=out_row, column=5, value=f"=IF({cmp_expr},{a_p},{b_p})")

    for c in range(1, 6):
        ws.column_dimensions[get_column_letter(c)].width = 16


def write_significance_count_sheet(wb, sheet_name, contrast_key, chromo_label, p, t, pids):
    """% of participants significant per channel (Fisher-combined, uncorrected);
    see the module docstring."""
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)

    headers = [
        "Channel",
        "N (both runs available)",
        "N significant (Fisher-combined, direction-checked)",
        "% significant",
        "N with conflicting direction (RunA vs RunB)",
        "N with unexpected direction (HbO decrease / HbR increase, both runs)",
    ]
    for col, h in enumerate(headers, start=1):
        ws.cell(row=1, column=col, value=h).font = bold

    run_a_p, run_b_p = p[contrast_key]["RunA"][chromo_label], p[contrast_key]["RunB"][chromo_label]
    run_a_t, run_b_t = t[contrast_key]["RunA"][chromo_label], t[contrast_key]["RunB"][chromo_label]
    exp_sign = expected_sign_for(chromo_label, contrast_key)

    for i, ch in enumerate(CHANNELS_OF_INTEREST):
        row = 2 + i
        ws.cell(row=row, column=1, value=f"CH {ch}").font = bold

        combined_p, status_by_pid = combined_pvalues_per_channel(
            run_a_p, run_b_p, run_a_t, run_b_t, ch, pids, expected_sign=exp_sign)
        n_valid = len(combined_p)
        n_sig = sum(1 for v in combined_p.values() if v < ALPHA)
        ws.cell(row=row, column=2, value=n_valid)
        ws.cell(row=row, column=3, value=n_sig)
        pct = ws.cell(row=row, column=4, value=(n_sig / n_valid) if n_valid else "")
        if n_valid:
            pct.number_format = "0.0%"
        ws.cell(row=row, column=5, value=sum(1 for s in status_by_pid.values() if s == "conflicting_direction"))
        ws.cell(row=row, column=6, value=sum(1 for s in status_by_pid.values() if s == "wrong_direction"))

    note_row = 2 + len(CHANNELS_OF_INTEREST) + 2
    ws.cell(row=note_row, column=1, value=(
        f"Fisher-combined: combined p-value of RunA+RunB (Fisher's method), only if both runs "
        f"agree in the sign of their T-values; otherwise the channel counts as not significant "
        f"for that participant (see column 'conflicting direction'). For Contrasts 1/2 the shared "
        f"direction must also match the expected one (HbO: T > 0, HbR: T < 0), see "
        f"REQUIRE_EXPECTED_DIRECTION in alpha_correction_utils.py. Significance: combined p < {ALPHA}, "
        f"uncorrected."
    )).font = Font(italic=True)

    for c in range(1, 7):
        ws.column_dimensions[get_column_letter(c)].width = 26


def main():
    parser = build_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pids = participant_ids(args.n_participants)
    p, t, missing_files, missing_sheets = load_glm_results(args.input_dir, args.n_participants)

    for contrast_key in CONTRASTS:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        run_sheets = {chromo: {} for chromo in CHROMOPHORES}
        for chromo in CHROMOPHORES:
            for run in RUNS:
                name = f"{run}_{chromo}"
                write_run_sheet(wb, name, t[contrast_key][run][chromo],
                                p[contrast_key][run][chromo], pids, chromo)
                run_sheets[chromo][run] = name

        for chromo in CHROMOPHORES:
            write_best_of_both_runs_sheet(
                wb, f"Best_{chromo}", run_sheets[chromo]["RunA"], run_sheets[chromo]["RunB"], pids, chromo)

        for chromo in CHROMOPHORES:
            write_significance_count_sheet(
                wb, f"SigCount_{chromo}", contrast_key, chromo, p, t, pids)

        out_path = args.output_dir / f"Summary_{contrast_key}.xlsx"
        wb.save(out_path)
        print(f"Saved: {out_path}")

    report_missing(missing_files, missing_sheets)


if __name__ == "__main__":
    main()
