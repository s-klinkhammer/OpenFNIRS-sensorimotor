"""
02 - Per-participant significance and the "gain from ID channels" analysis.

For every contrast one workbook (``Significance_Contrast1.xlsx``, ...) is
written with the sheets ``Oxy``, ``DeOxy`` and ``Combined_HbO_HbR_Union``.
Rows are participants, columns are channels.

Per-participant/channel p-value
-------------------------------
RunA and RunB are combined with Fisher's method (see
``alpha_correction_utils.py``). The runs are only combined if their T-values
agree in direction; for the "> Baseline" contrasts the shared direction must
also match the expected one (HbO: T > 0, HbR: T < 0). Otherwise the channel
counts as not significant for that participant (conflicting / unexpected
direction counts are reported in the sheet notes).

Sheets ``Oxy`` / ``DeOxy``: per-channel significance
----------------------------------------------------
One label column per channel: Fisher-combined p < ALPHA, uncorrected.

Sheet ``Combined_HbO_HbR_Union``: gain from the additional ID channels
----------------------------------------------------------------------
Two family-wise tests per participant:
  (a) at least one significant test among the SD & ID channels
      (what would have been found WITHOUT the extra channels);
  (b) at least one significant test among the ID-only channels
      (what the extra channels contribute).
Each family has 6 tests (3 channels x {HbO, HbR}); a family counts as
significant for a participant if ANY of its tests is significant. A participant
"gained activation via the ID channels" if (a) is false AND (b) is true.
Participants without any significant test do not count, because the ID
channels gave them nothing.

Correction: Bonferroni with family size M_eff (Li & Ji 2005), estimated once
per contrast and channel family from the 6x6 across-participant correlation of
the (run-averaged) T-values. A fixed family size of 6 is reported as a
robustness check.

Usage:
    python 02_significance_per_participant.py --input-dir <GLM results> --output-dir <out>
"""

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from alpha_correction_utils import bonferroni, effective_family_size, expected_sign_for
from glm_common import (
    ALPHA, CHANNELS_OF_INTEREST, CHROMOPHORES, CONTRASTS, ID_ONLY_CHANNELS,
    SD_AND_ID_CHANNELS, build_parser, combined_pvalues_per_channel,
    combined_tvalues_multi_chromo, load_glm_results, participant_ids, report_missing,
)

BOLD = Font(bold=True)
ITALIC = Font(italic=True)


def significance_label(combined_p, status, alpha=ALPHA):
    """Cell label for one participant/channel (from ``combine_two_run_pvalue_fisher``)."""
    if status == "missing":
        return "missing data"
    if status == "conflicting_direction":
        return "not significant (conflicting direction)"
    if status == "wrong_direction":
        return "not significant (unexpected direction)"
    return "significant" if combined_p < alpha else "not significant"


def count_significant(adjusted_p):
    """Number of entries of {key: adjusted p or None} below ALPHA."""
    return sum(1 for v in adjusted_p.values() if v is not None and v < ALPHA)


def gained_flag(any_sd_id_data, any_id_only_data, sd_id_sig, id_only_sig):
    """True if no SD&ID channel but at least one ID-only channel is
    significant; 'missing data' if either family has no data at all."""
    if not (any_sd_id_data and any_id_only_data):
        return "missing data"
    return (sd_id_sig == 0) and (id_only_sig > 0)


def write_significance_sheet(wb, sheet_name, p_runa, p_runb, t_runa, t_runb,
                             contrast_key, chromo_label, pids):
    """Sheets ``Oxy`` / ``DeOxy``: per-channel labels (Fisher-combined,
    uncorrected). Returns a note on conflicting / unexpected directions."""
    ws = wb.create_sheet(sheet_name)
    center = Alignment(horizontal="center")
    sig_fill = PatternFill("solid", start_color="C6EFCE", end_color="C6EFCE")
    exp_sign = expected_sign_for(chromo_label, contrast_key)

    combined_raw, combined_status = {}, {}
    for ch in CHANNELS_OF_INTEREST:
        combined_raw[ch], combined_status[ch] = combined_pvalues_per_channel(
            p_runa, p_runb, t_runa, t_runb, ch, pids, expected_sign=exp_sign)

    n_conflicting = {ch: sum(1 for s in combined_status[ch].values() if s == "conflicting_direction")
                     for ch in CHANNELS_OF_INTEREST}
    n_wrong = {ch: sum(1 for s in combined_status[ch].values() if s == "wrong_direction")
               for ch in CHANNELS_OF_INTEREST}

    ws.cell(row=1, column=1, value="Participant").font = BOLD
    for j, ch in enumerate(CHANNELS_OF_INTEREST):
        cell = ws.cell(row=1, column=2 + j, value=f"CH {ch} (Fisher-combined, direction-checked)")
        cell.font, cell.alignment = BOLD, center

    for offset, pid in enumerate(pids):
        row_idx = offset + 2
        ws.cell(row=row_idx, column=1, value=pid).font = BOLD
        for j, ch in enumerate(CHANNELS_OF_INTEREST):
            label = significance_label(combined_raw[ch].get(pid), combined_status[ch].get(pid))
            cell = ws.cell(row=row_idx, column=2 + j, value=label)
            cell.alignment = center
            if label == "significant":
                cell.fill = sig_fill

    ws.column_dimensions["A"].width = 12
    for c in range(2, 2 + len(CHANNELS_OF_INTEREST)):
        ws.column_dimensions[get_column_letter(c)].width = 30

    return (
        "Participants with RunA/RunB pointing in conflicting directions (not combined, forced to "
        "'not significant'), per channel: "
        + ", ".join(f"CH {ch}: n={n_conflicting[ch]}" for ch in CHANNELS_OF_INTEREST)
        + ". Participants with both runs in the UNEXPECTED direction (HbO decrease / HbR increase; "
        "only checked for Contrasts 1/2; forced to 'not significant'): "
        + ", ".join(f"CH {ch}: n={n_wrong[ch]}" for ch in CHANNELS_OF_INTEREST)
    )


def add_summary_rows(ws, flag_cols, notes, family_label, fam_keys, pids):
    """Append '% participants who gained activation via ID channels' rows
    (one per correction), followed by explanatory notes.

    ``fam_keys`` maps flag-column keys to labels, e.g. {"raw": "raw", ...}.
    The percentages are computed in Python from the static flag columns."""
    first_row, last_row = 2, len(pids) + 1
    summary_row = last_row + 2
    for key, label in fam_keys.items():
        col = flag_cols[key]
        values = [ws.cell(row=r, column=col).value for r in range(first_row, last_row + 1)]
        valid = [v for v in values if v != "missing data"]
        n_gained = sum(1 for v in valid if v is True)
        ws.cell(row=summary_row, column=1,
                value=f"% participants gained activation via ID channels{family_label} - {label}").font = BOLD
        pct_cell = ws.cell(row=summary_row, column=col, value=(n_gained / len(valid)) if valid else "")
        if valid:
            pct_cell.number_format = "0.0%"
        summary_row += 1

    note_row = summary_row + 1
    for k, note in enumerate(notes):
        ws.cell(row=note_row + k, column=1, value=note).font = ITALIC


def write_combined_union_sheet(wb, p_by_chromo_runa, p_by_chromo_runb,
                               t_by_chromo_runa, t_by_chromo_runb, contrast_key, pids):
    """Sheet ``Combined_HbO_HbR_Union``; returns (flag_cols, notes)."""
    ws = wb.create_sheet("Combined_HbO_HbR_Union")

    combined_raw, combined_status = {}, {}   # keyed by (channel, chromophore)
    for chromo in CHROMOPHORES:
        for ch in CHANNELS_OF_INTEREST:
            combined_raw[(ch, chromo)], combined_status[(ch, chromo)] = combined_pvalues_per_channel(
                p_by_chromo_runa[chromo], p_by_chromo_runb[chromo],
                t_by_chromo_runa[chromo], t_by_chromo_runb[chromo], ch, pids,
                expected_sign=expected_sign_for(chromo, contrast_key))

    sd_id_virtual = [(ch, c) for ch in SD_AND_ID_CHANNELS for c in CHROMOPHORES]
    id_only_virtual = [(ch, c) for ch in ID_ONLY_CHANNELS for c in CHROMOPHORES]

    meff_sd_id, n_used_sd_id = effective_family_size(
        combined_tvalues_multi_chromo(t_by_chromo_runa, t_by_chromo_runb, SD_AND_ID_CHANNELS, pids),
        sd_id_virtual)
    meff_id_only, n_used_id_only = effective_family_size(
        combined_tvalues_multi_chromo(t_by_chromo_runa, t_by_chromo_runb, ID_ONLY_CHANNELS, pids),
        id_only_virtual)

    id_only_cols = {"meff": 2, "bonf6": 3}
    sd_id_cols = {"meff": 4, "bonf6": 5}
    flag_cols = {"meff": 6, "bonf6": 7}

    ws.cell(row=1, column=1, value="Participant").font = BOLD
    headers = {
        id_only_cols["meff"]: f"ID only significant, Bonferroni fam=M_eff={meff_id_only:.0f} (n of 6)",
        id_only_cols["bonf6"]: "ID only significant, Bonferroni fam=6 (n of 6; robustness)",
        sd_id_cols["meff"]: f"SD & ID significant, Bonferroni fam=M_eff={meff_sd_id:.0f} (n of 6)",
        sd_id_cols["bonf6"]: "SD & ID significant, Bonferroni fam=6 (n of 6; robustness)",
        flag_cols["meff"]: "Gained activation via ID channels - Bonferroni fam=M_eff (SD&ID=0 AND ID-only>0)",
        flag_cols["bonf6"]: "Gained activation via ID channels - Bonferroni fam=6 (robustness)",
    }
    for col, h in headers.items():
        ws.cell(row=1, column=col, value=h).font = BOLD

    for offset, pid in enumerate(pids):
        row_idx = offset + 2
        ws.cell(row=row_idx, column=1, value=pid).font = BOLD

        sd_id_p = {vc: combined_raw[vc][pid] for vc in sd_id_virtual if pid in combined_raw[vc]}
        id_only_p = {vc: combined_raw[vc][pid] for vc in id_only_virtual if pid in combined_raw[vc]}

        sd_id_sig = {
            "meff": count_significant(bonferroni(sd_id_p, family_size=meff_sd_id)),
            "bonf6": count_significant(bonferroni(sd_id_p, family_size=6)),
        }
        id_only_sig = {
            "meff": count_significant(bonferroni(id_only_p, family_size=meff_id_only)),
            "bonf6": count_significant(bonferroni(id_only_p, family_size=6)),
        }
        for key in ("meff", "bonf6"):
            ws.cell(row=row_idx, column=id_only_cols[key], value=id_only_sig[key])
            ws.cell(row=row_idx, column=sd_id_cols[key], value=sd_id_sig[key])
            ws.cell(row=row_idx, column=flag_cols[key],
                    value=gained_flag(len(sd_id_p) > 0, len(id_only_p) > 0,
                                      sd_id_sig[key], id_only_sig[key]))

    for c in range(1, 8):
        ws.column_dimensions[get_column_letter(c)].width = 30

    n_conflicting = {vc: sum(1 for s in combined_status[vc].values() if s == "conflicting_direction")
                     for vc in sd_id_virtual + id_only_virtual}
    notes = [
        "A channel family counts as significant for a participant as soon as ANY of its 3 channels is "
        "significant in EITHER chromophore (HbO or HbR; 6 tests per family). Per (channel, chromophore): "
        "Fisher combination of RunA/RunB, only if both runs agree in direction (for Contrasts 1/2 also "
        "matching the expected direction); Bonferroni correction per participant, threshold p < "
        f"{ALPHA} after correction.",
        f"M_eff (Li & Ji 2005), estimated from the 6x6 across-participant correlation of the run-averaged "
        f"T-values: SD&ID -> M_eff={meff_sd_id:.0f} (n={n_used_sd_id} complete cases); "
        f"ID-only -> M_eff={meff_id_only:.0f} (n={n_used_id_only} complete cases). "
        f"M_eff=1: fully correlated; M_eff=6: independent (identical to the fam=6 column).",
        "Participant/(channel, chromophore) combinations with RunA/RunB in conflicting directions "
        "(not combined, forced to 'not significant'): "
        + ", ".join(f"CH {ch} {chromo}: n={n_conflicting[(ch, chromo)]}"
                    for ch in CHANNELS_OF_INTEREST for chromo in CHROMOPHORES),
    ]
    return flag_cols, notes


def main():
    parser = build_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pids = participant_ids(args.n_participants)
    p, t, missing_files, missing_sheets = load_glm_results(args.input_dir, args.n_participants)

    for contrast_key in CONTRASTS:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        for chromo in CHROMOPHORES:
            conflicting_note = write_significance_sheet(
                wb, chromo,
                p[contrast_key]["RunA"][chromo], p[contrast_key]["RunB"][chromo],
                t[contrast_key]["RunA"][chromo], t[contrast_key]["RunB"][chromo],
                contrast_key, chromo, pids)
            ws = wb[chromo]
            ws.cell(row=len(pids) + 3, column=1, value=(
                f"Fisher's method on RunA/RunB, only if both runs agree in the sign of their T-values, "
                f"otherwise 'not significant'. Significance: combined p < {ALPHA}, uncorrected.")).font = ITALIC
            ws.cell(row=len(pids) + 4, column=1, value=conflicting_note).font = ITALIC

        flag_cols, notes = write_combined_union_sheet(
            wb, p[contrast_key]["RunA"], p[contrast_key]["RunB"],
            t[contrast_key]["RunA"], t[contrast_key]["RunB"], contrast_key, pids)
        add_summary_rows(
            wb["Combined_HbO_HbR_Union"], flag_cols, notes, family_label=" (HbO/HbR union)",
            fam_keys={"meff": "Bonferroni fam=M_eff", "bonf6": "Bonferroni fam=6"},
            pids=pids)

        out_path = args.output_dir / f"Significance_{contrast_key}.xlsx"
        wb.save(out_path)
        print(f"Saved: {out_path}")

    report_missing(missing_files, missing_sheets)


if __name__ == "__main__":
    main()
