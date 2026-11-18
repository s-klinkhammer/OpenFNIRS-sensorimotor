"""
04 - Generate the results text and tables for the paper.

Reads the per-participant GLM result files and writes three result blocks as
a plain-text/Markdown file (``paper_results_<timestamp>.txt``) and, if the
``python-docx`` package is installed, as a Word file with real tables
(``paper_results_<timestamp>.docx``).

The statistics are the same as in scripts 01 and 02 and use the same
functions from ``alpha_correction_utils.py``; this script only condenses them
into ready-to-use sentences and tables.

Result blocks
-------------
(1) Percentage of participants with significant activation per channel
    (Hand clenching / Finger tapping vs. baseline; HbO and HbR).
    Fisher-combined p-value of RunA+RunB, direction-checked, p < ALPHA.
    Descriptive, so no multiple-comparison correction.
(2) Percentage of participants for whom each channel was the "best channel":
    the channel with the strongest activation in the expected direction
    (HbO: highest T, HbR: most negative T) over both runs, for all three
    contrasts.
(3) Participants who would have shown no significant activation without the
    ID-only channels ("non-responders"), and how many of them are "rescued"
    by the ID-only channels. Each channel family (SD & ID / ID-only) has 6
    tests (3 channels x HbO/HbR) and counts as significant if any of them is.
    Family-wise correction per participant: Bonferroni with family size M_eff
    (Li & Ji 2005), estimated from the 6x6 across-participant correlation
    matrix of the run-averaged T-values. A fixed family size of 6 is reported
    as a robustness check.

BEFORE RUNNING: check ``CHANNEL_LABELS`` below. It maps the channel numbers in
the raw data to the optode-pair labels used in the paper and must match your
montage.

Usage:
    python 04_generate_paper_results.py --input-dir <GLM results> --output-dir <out>
"""

import datetime

from alpha_correction_utils import (
    EXPECTED_T_SIGN, bonferroni, effective_family_size, expected_sign_for,
)
from glm_common import (
    ALPHA, CHANNELS_OF_INTEREST, CHROMOPHORE_PAPER_LABEL, CHROMOPHORES,
    CONTRASTS, CONTRASTS_VS_BASELINE, CONTRAST_TASK_LABEL, ID_ONLY_CHANNELS,
    SD_AND_ID_CHANNELS, build_parser, combined_pvalues_per_channel,
    combined_tvalues_multi_chromo, load_glm_results, participant_ids,
)

try:  # Word output is optional
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt
except ImportError:
    Document = None

# Channel number in the raw data -> optode-pair label (source-detector) used in
# the paper. ADAPT TO YOUR MONTAGE.
CHANNEL_LABELS = {
    4: "S2-D4",
    10: "S4-D4",
    13: "S5-D4",
    2: "S1-D5",
    6: "S3-D5",
    16: "S6-D5",
}

CONTRAST_PAPER_LABEL = {
    "Contrast1": "Hand vs BL",
    "Contrast2": "Finger vs BL",
    "Contrast3": "Hand vs Finger",
}


def ch_label(ch):
    return CHANNEL_LABELS.get(ch, f"CH {ch}")


# =============================================================================
# (1) % participants with significant activation, per channel (uncorrected)
# =============================================================================
def analysis_1(p, t, pids):
    """{contrast: {chromophore: {channel: {n, n_sig, pct, n_conflicting,
    n_wrong_direction}}}}"""
    results = {}
    for contrast_key in CONTRASTS_VS_BASELINE:
        results[contrast_key] = {}
        for chromo in CHROMOPHORES:
            p_a, p_b = p[contrast_key]["RunA"][chromo], p[contrast_key]["RunB"][chromo]
            t_a, t_b = t[contrast_key]["RunA"][chromo], t[contrast_key]["RunB"][chromo]
            exp_sign = expected_sign_for(chromo, contrast_key)

            per_channel = {}
            for ch in CHANNELS_OF_INTEREST:
                combined, status = combined_pvalues_per_channel(
                    p_a, p_b, t_a, t_b, ch, pids, expected_sign=exp_sign)
                n = sum(1 for s in status.values() if s != "missing")
                n_sig = sum(1 for v in combined.values() if v < ALPHA)
                per_channel[ch] = {
                    "n": n,
                    "n_sig": n_sig,
                    "pct": (n_sig / n) if n else None,
                    "n_conflicting": sum(1 for s in status.values() if s == "conflicting_direction"),
                    "n_wrong_direction": sum(1 for s in status.values() if s == "wrong_direction"),
                }
            results[contrast_key][chromo] = per_channel
    return results


# =============================================================================
# (2) "Best channel" per participant
# =============================================================================
def best_channel_per_participant(t_runa, t_runb, chromo_label, pids):
    """
    {pid: best_channel or None}.

    T-values are multiplied by the expected sign (+1 HbO, -1 HbR) so that
    "larger" always means "stronger activation in the expected direction".
    Within each run the channel with the highest direction-adjusted T is
    taken; across runs, the run with the larger adjusted peak decides (RunA on
    ties). Mirrors the ``Best_<chromophore>`` sheets of script 01.
    """
    sign = EXPECTED_T_SIGN[chromo_label]
    best = {}
    for pid in pids:
        def best_in_run(t_data):
            vals = {ch: t_data.get(pid, {}).get(ch) for ch in CHANNELS_OF_INTEREST}
            vals = {ch: sign * v for ch, v in vals.items() if v is not None}
            if not vals:
                return None, None
            ch_best = max(vals, key=lambda ch: vals[ch])
            return ch_best, vals[ch_best]

        ch_a, peak_a = best_in_run(t_runa)
        ch_b, peak_b = best_in_run(t_runb)
        if peak_a is None and peak_b is None:
            best[pid] = None
        elif peak_b is None:
            best[pid] = ch_a
        elif peak_a is None:
            best[pid] = ch_b
        else:
            best[pid] = ch_a if peak_a >= peak_b else ch_b
    return best


def analysis_2(t, pids):
    """{contrast: {chromophore: {"per_channel": {ch: share}, "total_id_only",
    "total_sd_id", "n_valid"}}}"""
    results = {}
    for contrast_key in CONTRASTS:
        results[contrast_key] = {}
        for chromo in CHROMOPHORES:
            best = best_channel_per_participant(
                t[contrast_key]["RunA"][chromo], t[contrast_key]["RunB"][chromo], chromo, pids)
            n_valid = sum(1 for v in best.values() if v is not None)
            share = {ch: (sum(1 for v in best.values() if v == ch) / n_valid) if n_valid else None
                     for ch in CHANNELS_OF_INTEREST}
            results[contrast_key][chromo] = {
                "per_channel": share,
                "total_id_only": sum(share[ch] for ch in ID_ONLY_CHANNELS) if n_valid else None,
                "total_sd_id": sum(share[ch] for ch in SD_AND_ID_CHANNELS) if n_valid else None,
                "n_valid": n_valid,
            }
    return results


# =============================================================================
# (3) Non-responders without ID-only channels, and rescue by ID-only channels
# =============================================================================
def _family_significant(raw_p, family_size):
    """True if any channel of the family is significant after Bonferroni."""
    adjusted = bonferroni(raw_p, family_size=family_size)
    return any(v is not None and v < ALPHA for v in adjusted.values())


def _tally(counts, sd_id_sig, id_only_sig):
    """Update the non-responder counters for one participant.
    no_sig_sd_id: no SD&ID channel significant (non-responder without ID-only);
    rescued:      ... but at least one ID-only channel significant;
    no_sig_all:   ... and no ID-only channel significant either."""
    if not sd_id_sig:
        counts["no_sig_sd_id"] += 1
        counts["rescued" if id_only_sig else "no_sig_all"] += 1


def _summarise(counts, n_valid, suffix=""):
    out = {}
    for key in ("no_sig_sd_id", "rescued", "no_sig_all"):
        out[f"n_{key}{suffix}"] = counts[key]
        out[f"pct_{key}{suffix}"] = (counts[key] / n_valid) if n_valid else None
    return out


def _new_counts():
    return {"no_sig_sd_id": 0, "rescued": 0, "no_sig_all": 0}


def analysis_3(p, t, pids):
    """
    Non-responders and rescue by ID-only channels. A family (SD & ID / ID-only)
    is significant for a participant if any of its 3 channels is significant
    in either chromophore (6 tests per family). M_eff is estimated from the
    6x6 T-value correlation matrix; fixed family size 6 ("_fam6") is the
    robustness check. ``n_valid`` = participants with at least one non-missing
    SD&ID test. Returns {contrast: {n_/pct_ no_sig_sd_id, rescued, no_sig_all,
    (same with suffix _fam6), n_valid, meff_*, n_used_*, n_conflicting}}.
    """
    results = {}
    for contrast_key in CONTRASTS_VS_BASELINE:
        combined_raw, combined_status = {}, {}   # keyed by (channel, chromophore)
        for chromo in CHROMOPHORES:
            for ch in CHANNELS_OF_INTEREST:
                combined_raw[(ch, chromo)], combined_status[(ch, chromo)] = combined_pvalues_per_channel(
                    p[contrast_key]["RunA"][chromo], p[contrast_key]["RunB"][chromo],
                    t[contrast_key]["RunA"][chromo], t[contrast_key]["RunB"][chromo], ch, pids,
                    expected_sign=expected_sign_for(chromo, contrast_key))

        sd_id_virtual = [(ch, c) for ch in SD_AND_ID_CHANNELS for c in CHROMOPHORES]
        id_only_virtual = [(ch, c) for ch in ID_ONLY_CHANNELS for c in CHROMOPHORES]
        t_runa, t_runb = t[contrast_key]["RunA"], t[contrast_key]["RunB"]
        meff_sd_id, n_used_sd_id = effective_family_size(
            combined_tvalues_multi_chromo(t_runa, t_runb, SD_AND_ID_CHANNELS, pids), sd_id_virtual)
        meff_id_only, n_used_id_only = effective_family_size(
            combined_tvalues_multi_chromo(t_runa, t_runb, ID_ONLY_CHANNELS, pids), id_only_virtual)

        n_valid = 0
        counts, counts_fam6 = _new_counts(), _new_counts()
        for pid in pids:
            sd_id_p = {vc: combined_raw[vc].get(pid) for vc in sd_id_virtual}
            id_only_p = {vc: combined_raw[vc].get(pid) for vc in id_only_virtual}
            if all(v is None for v in sd_id_p.values()):
                continue
            n_valid += 1
            _tally(counts, _family_significant(sd_id_p, meff_sd_id),
                   _family_significant(id_only_p, meff_id_only))
            _tally(counts_fam6, _family_significant(sd_id_p, 6), _family_significant(id_only_p, 6))

        res = {"n_valid": n_valid,
               "meff_sd_id": meff_sd_id, "n_used_sd_id": n_used_sd_id,
               "meff_id_only": meff_id_only, "n_used_id_only": n_used_id_only,
               "n_conflicting": sum(1 for vc in sd_id_virtual + id_only_virtual
                                    for s in combined_status[vc].values()
                                    if s == "conflicting_direction")}
        res.update(_summarise(counts, n_valid))
        res.update(_summarise(counts_fam6, n_valid, "_fam6"))
        results[contrast_key] = res
    return results


def meff_summary_lines(res3):
    """One line per contrast with the M_eff values of block (3)."""
    return [
        f"{CONTRAST_TASK_LABEL[c]}: M_eff(SD&ID)={res3[c]['meff_sd_id']:.0f} (n={res3[c]['n_used_sd_id']}), "
        f"M_eff(ID-only)={res3[c]['meff_id_only']:.0f} (n={res3[c]['n_used_id_only']})"
        for c in CONTRASTS_VS_BASELINE]


# =============================================================================
# Text formatting
# =============================================================================
def fmt_pct(x, decimals=0):
    return "n/a" if x is None else f"{x * 100:.{decimals}f}%"


def fmt_n_pct(res, key, decimals=1):
    """'n (xx.x%)' for key in {'no_sig_sd_id', 'rescued', 'no_sig_all', ...}."""
    return f"{res['n_' + key]} ({fmt_pct(res['pct_' + key], decimals)})"


def most_consistent_id_only_channel(res1, contrast_key, chromo, pct_key):
    """(channel, share) of the ID-only channel with the highest share of
    significant participants, or (None, None)."""
    shares = {ch: res1[contrast_key][chromo][ch][pct_key] for ch in ID_ONLY_CHANNELS}
    candidates = [ch for ch in shares if shares[ch] is not None]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda ch: shares[ch])
    return best, shares[best]


def most_consistent_lines(res1, pct_key):
    lines = []
    for contrast_key in CONTRASTS_VS_BASELINE:
        for chromo in CHROMOPHORES:
            ch, share = most_consistent_id_only_channel(res1, contrast_key, chromo, pct_key)
            if ch is not None:
                lines.append(
                    f"{CONTRAST_TASK_LABEL[contrast_key]}, {CHROMOPHORE_PAPER_LABEL[chromo]}: "
                    f"most consistently significant ID-only channel = {ch_label(ch)} ({fmt_pct(share)})")
    return lines


def non_responder_paragraph(res3):
    """Ready-to-use results paragraph for block (3)."""
    c1, c2 = res3["Contrast1"], res3["Contrast2"]
    return (
        f"Without the additional ID channels, {fmt_pct(c1['pct_no_sig_sd_id'], 1)} of participants "
        f"(n = {c1['n_no_sig_sd_id']}) showed no significant task-related activation in either HbO "
        f"or HbR (Fisher's combined p, Bonferroni-corrected with M_eff; alpha = {ALPHA}) during Hand "
        f"Clenching, and {fmt_pct(c2['pct_no_sig_sd_id'], 1)} (n = {c2['n_no_sig_sd_id']}) during "
        f"Finger Tapping. Once the ID-only channels were included, these non-responder rates "
        f"dropped to {fmt_pct(c1['pct_no_sig_all'], 1)} (n = {c1['n_no_sig_all']}) for Hand Clenching "
        f"and {fmt_pct(c2['pct_no_sig_all'], 1)} (n = {c2['n_no_sig_all']}) for Finger Tapping, "
        f"corresponding to reductions of {fmt_pct(c1['pct_rescued'], 1)} (n = {c1['n_rescued']}) "
        f"and {fmt_pct(c2['pct_rescued'], 1)} (n = {c2['n_rescued']}), respectively."
    )


BLOCK_3_METHOD_NOTE = (
    "Per-participant/channel p-value: Fisher's combined p-value of RunA+RunB, direction-checked "
    "(forced to p=1.0, i.e. not significant, if RunA and RunB disagree in the sign of their "
    "T-values or the shared direction is not the expected one; see "
    "combine_two_run_pvalue_fisher() in alpha_correction_utils.py). A channel family (SD & ID / "
    "ID-only) counts as significant for a participant if ANY of its 3 channels is significant in "
    "EITHER HbO or HbR (6 tests per family). Correction: per participant, Bonferroni with family "
    "size M_eff (Li & Ji 2005), estimated from the 6x6 cross-participant correlation matrix of the "
    "run-averaged T-values. A fixed family size of 6 is shown as a robustness check only."
)


def build_block_1(res1):
    lines = ["(1) Percentage of participants showing significant activation per channel", "",
             "Fisher-combined p-value of RunA+RunB, direction-checked, uncorrected (p < "
             f"{ALPHA}).", ""]
    for contrast_key in CONTRASTS_VS_BASELINE:
        lines += [f"## {CONTRAST_TASK_LABEL[contrast_key]} vs. Baseline",
                  "| Group | Channel | HbO | HbR |",
                  "|---|---|---|---|"]
        for group, channels in (("SD & ID", SD_AND_ID_CHANNELS), ("ID only", ID_ONLY_CHANNELS)):
            for ch in channels:
                o, d = res1[contrast_key]["Oxy"][ch], res1[contrast_key]["DeOxy"][ch]
                lines.append(f"| {group} | {ch_label(ch)} | {fmt_pct(o['pct'])} | {fmt_pct(d['pct'])} |")
        lines.append("")
    lines += ["Text snippet:", ""]
    lines += ["  " + s for s in most_consistent_lines(res1, "pct")]
    lines.append("")
    return "\n".join(lines)


def build_block_2(res2):
    ids = [ch_label(ch) for ch in ID_ONLY_CHANNELS]
    sds = [ch_label(ch) for ch in SD_AND_ID_CHANNELS]
    lines = ['(2) Percentage of participants for whom a given channel was the "best channel"', "",
             "Table X:", "",
             "|    |    | " + " | ".join(["ID only channels"] * len(ids)) + " | | | "
             + " | ".join(["SD & ID channels"] * len(sds)) + " | |",
             "|    |    | " + " | ".join(ids) + " | Total % |    | " + " | ".join(sds) + " | Total % |",
             "|---" * (5 + len(ids) + len(sds)) + "|"]
    for chromo in CHROMOPHORES:
        for j, contrast_key in enumerate(CONTRASTS):
            row = res2[contrast_key][chromo]
            lines.append(
                f"| {CHROMOPHORE_PAPER_LABEL[chromo] if j == 0 else ''} | {CONTRAST_PAPER_LABEL[contrast_key]} | "
                + " | ".join(fmt_pct(row["per_channel"][ch]) for ch in ID_ONLY_CHANNELS)
                + f" | {fmt_pct(row['total_id_only'])} | | "
                + " | ".join(fmt_pct(row["per_channel"][ch]) for ch in SD_AND_ID_CHANNELS)
                + f" | {fmt_pct(row['total_sd_id'])} |")
    lines.append("")
    return "\n".join(lines)


def build_block_3(res3):
    lines = ["(3) Participants who would have lacked any significant activation without the ID channels",
             "", BLOCK_3_METHOD_NOTE, ""]
    for contrast_key in CONTRASTS_VS_BASELINE:
        rc = res3[contrast_key]
        lines += [
            f"{CONTRAST_TASK_LABEL[contrast_key]} (HbO+HbR):",
            f"  (1) Non-responders, SD&ID channels only: {fmt_n_pct(rc, 'no_sig_sd_id')} of {rc['n_valid']}",
            f"  (2) Of those, rescued by ID-only channels: {fmt_n_pct(rc, 'rescued')}",
            f"  (3) Still non-responders with ID-only channels added: {fmt_n_pct(rc, 'no_sig_all')}",
            f"  M_eff: SD&ID={rc['meff_sd_id']:.0f} (n={rc['n_used_sd_id']}), "
            f"ID-only={rc['meff_id_only']:.0f} (n={rc['n_used_id_only']})",
            f"  [robustness, fixed fam=6] baseline: {fmt_n_pct(rc, 'no_sig_sd_id_fam6')}; "
            f"rescued: {fmt_n_pct(rc, 'rescued_fam6')}; still non-resp.: {fmt_n_pct(rc, 'no_sig_all_fam6')}",
            f"  n (participant, channel, chromophore) with conflicting RunA/RunB direction "
            f"(not combined, forced to 'not significant'): {rc['n_conflicting']}",
            "",
        ]
    lines += ["Text snippet:", "", non_responder_paragraph(res3), ""]
    return "\n".join(lines)


# =============================================================================
# Word (.docx) output
# =============================================================================
def _set_cell_text(cell, text, bold=False, size=10):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)


def _shade_cell(cell, hex_color="D9D9D9"):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(tc_pr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): hex_color}))


def _add_table(doc, headers, rows):
    """Table with a shaded bold header row; ``rows`` is a list of lists of str."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, h in zip(table.rows[0].cells, headers):
        _set_cell_text(cell, h, bold=True)
        _shade_cell(cell)
    for values in rows:
        for cell, v in zip(table.add_row().cells, values):
            _set_cell_text(cell, v)
    return table


def _add_bullets(doc, lines):
    for line in lines:
        doc.add_paragraph(line, style="List Bullet")


def add_docx_block_1(doc, res1):
    doc.add_heading("(1) Percentage of participants showing significant activation per channel", level=2)
    doc.add_paragraph(
        f"Fisher-combined p-value of RunA+RunB, direction-checked, uncorrected (p < {ALPHA}).")
    for contrast_key in CONTRASTS_VS_BASELINE:
        doc.add_heading(f"{CONTRAST_TASK_LABEL[contrast_key]} vs. Baseline", level=3)
        rows = []
        for group, channels in (("SD & ID", SD_AND_ID_CHANNELS), ("ID only", ID_ONLY_CHANNELS)):
            for ch in channels:
                o, d = res1[contrast_key]["Oxy"][ch], res1[contrast_key]["DeOxy"][ch]
                rows.append([group, ch_label(ch), fmt_pct(o["pct"]), fmt_pct(d["pct"])])
        _add_table(doc, ["Group", "Channel", "HbO", "HbR"], rows)
        doc.add_paragraph("")
    doc.add_heading("Text snippet", level=3)
    _add_bullets(doc, most_consistent_lines(res1, "pct"))


def add_docx_block_2(doc, res2):
    doc.add_heading('(2) Percentage of participants for whom a given channel was the "best channel"', level=2)
    doc.add_paragraph("Table X")
    n_id, n_sd = len(ID_ONLY_CHANNELS), len(SD_AND_ID_CHANNELS)
    table = doc.add_table(rows=2, cols=2 + n_id + 1 + n_sd + 1)
    table.style = "Table Grid"
    id_start, id_end = 2, 2 + n_id - 1
    sd_start, sd_end = id_end + 2, id_end + 2 + n_sd - 1

    top = table.rows[0].cells
    id_merged, sd_merged = top[id_start].merge(top[id_end]), top[sd_start].merge(top[sd_end])
    _set_cell_text(id_merged, "ID only channels", bold=True)
    _set_cell_text(sd_merged, "SD & ID channels", bold=True)
    for cell in (id_merged, sd_merged):
        _shade_cell(cell)
    for c in (0, 1, id_end + 1, sd_end + 1):
        _set_cell_text(top[c], "")
        _shade_cell(top[c])

    header = table.rows[1].cells
    for c in (0, 1):
        _set_cell_text(header[c], "", bold=True)
    for i, ch in enumerate(ID_ONLY_CHANNELS):
        _set_cell_text(header[id_start + i], ch_label(ch), bold=True)
    _set_cell_text(header[id_end + 1], "Total %", bold=True)
    for i, ch in enumerate(SD_AND_ID_CHANNELS):
        _set_cell_text(header[sd_start + i], ch_label(ch), bold=True)
    _set_cell_text(header[sd_end + 1], "Total %", bold=True)
    for cell in header:
        _shade_cell(cell)

    for chromo in CHROMOPHORES:
        for j, contrast_key in enumerate(CONTRASTS):
            row = table.add_row().cells
            data = res2[contrast_key][chromo]
            _set_cell_text(row[0], CHROMOPHORE_PAPER_LABEL[chromo] if j == 0 else "", bold=(j == 0))
            _set_cell_text(row[1], CONTRAST_PAPER_LABEL[contrast_key])
            for i, ch in enumerate(ID_ONLY_CHANNELS):
                _set_cell_text(row[id_start + i], fmt_pct(data["per_channel"][ch]))
            _set_cell_text(row[id_end + 1], fmt_pct(data["total_id_only"]))
            for i, ch in enumerate(SD_AND_ID_CHANNELS):
                _set_cell_text(row[sd_start + i], fmt_pct(data["per_channel"][ch]))
            _set_cell_text(row[sd_end + 1], fmt_pct(data["total_sd_id"]))
    doc.add_paragraph("")


def add_docx_block_3(doc, res3):
    doc.add_heading("(3) Participants who would have lacked any significant activation without the ID channels", level=2)
    doc.add_paragraph(BLOCK_3_METHOD_NOTE)
    doc.add_paragraph(non_responder_paragraph(res3))

    headers = ["Task", "Non-resp. SD&ID (n, %)", "Rescued by ID-only (n, %)",
               "Still non-resp. (n, %)", "n valid"]

    def rows_for(suffix):
        return [[CONTRAST_TASK_LABEL[c], fmt_n_pct(res3[c], "no_sig_sd_id" + suffix),
                 fmt_n_pct(res3[c], "rescued" + suffix), fmt_n_pct(res3[c], "no_sig_all" + suffix),
                 str(res3[c]["n_valid"])] for c in CONTRASTS_VS_BASELINE]

    doc.add_heading("Non-responder breakdown (M_eff-corrected)", level=3)
    _add_table(doc, headers, rows_for(""))
    doc.add_heading("Robustness: fixed family size 6", level=3)
    _add_table(doc, headers, rows_for("_fam6"))
    doc.add_heading("M_eff values used above (Li & Ji 2005)", level=3)
    _add_bullets(doc, meff_summary_lines(res3))
    doc.add_heading("RunA/RunB direction conflicts (not combined)", level=3)
    _add_bullets(doc, [
        f"{CONTRAST_TASK_LABEL[c]}: n={res3[c]['n_conflicting']} participant/channel/chromophore "
        f"combinations where RunA and RunB pointed in opposite directions "
        f"(forced to 'not significant' rather than combined)"
        for c in CONTRASTS_VS_BASELINE])


def build_docx(res1, res2, res3, out_path):
    doc = Document()
    doc.add_heading("Auto-generated results text - check CHANNEL_LABELS before use", level=1)
    add_docx_block_1(doc, res1)
    doc.add_page_break()
    add_docx_block_2(doc, res2)
    doc.add_page_break()
    add_docx_block_3(doc, res3)
    doc.save(out_path)


def main():
    parser = build_parser(__doc__.split("\n\n")[0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pids = participant_ids(args.n_participants)
    print("Reading GLM result files ...")
    p, t, missing_files, missing_sheets = load_glm_results(args.input_dir, args.n_participants)

    print("Computing (1) per-channel significance ...")
    res1 = analysis_1(p, t, pids)
    print("Computing (2) best channel per participant ...")
    res2 = analysis_2(t, pids)
    print("Computing (3) non-responders and rescue by ID-only channels ...")
    res3 = analysis_3(p, t, pids)

    separator = "=" * 80
    parts = ["Auto-generated from raw GLM data - CHECK the CHANNEL_LABELS mapping before use.",
             separator, "", build_block_1(res1),
             separator, "", build_block_2(res2),
             separator, "", build_block_3(res3)]
    if missing_files:
        parts += [separator, f"{len(missing_files)} expected file(s) not found:"]
        parts += [f"  - {m}" for m in missing_files]
    if missing_sheets:
        parts += [f"{len(missing_sheets)} missing sheet(s) in existing files:"]
        parts += [f"  - {m}" for m in missing_sheets]
    text = "\n".join(parts)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    txt_path = args.output_dir / f"paper_results_{timestamp}.txt"
    txt_path.write_text(text, encoding="utf-8")
    print(f"\nSaved: {txt_path}")

    if Document is not None:
        docx_path = args.output_dir / f"paper_results_{timestamp}.docx"
        build_docx(res1, res2, res3, docx_path)
        print(f"Saved: {docx_path}")
    else:
        print("python-docx not installed - skipping the .docx output (pip install python-docx).")

    print("\n" + text)


if __name__ == "__main__":
    main()
