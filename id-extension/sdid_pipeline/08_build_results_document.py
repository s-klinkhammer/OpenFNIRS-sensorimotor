"""
08_build_results_document.py
Builds ONE Word document with the complete results summary from the outputs
of 02_nested_cv.py, 03_stats_report.py, 04_group_vs_chance.py,
06_quality_channel_labels.py and 07_quality_group_stats.py.

Structure
  - Multivariate results (group and participant level): full detail
  - Data-quality sensitivity analysis: full detail
  - Shared latent space sensitivity analysis: condensed in the main text,
    full statistics in the appendix

Nothing is recomputed: the script only reads existing CSV/XLSX files, so it
must run after steps 02-04 (both modes) and 06-07. Requires python-docx.
"""

import os
import glob
from datetime import datetime

import pandas as pd
from docx import Document
from docx.shared import Inches

import config
from core.stats_core import effect_size_words


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def load_condition_summary(output_dir, prefix, cond):
    path = os.path.join(output_dir, f"{prefix}_{cond.lower()}_individual_vs_chance.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_overlap(output_dir, prefix):
    """Read {prefix}_sd_id_overlap.csv (from 04_group_vs_chance.py) and return the four
    contingency cells plus derived quantities, or None if the file is missing."""
    path = os.path.join(output_dir, f"{prefix}_sd_id_overlap.csv")
    if not os.path.exists(path):
        return None
    ct = pd.read_csv(path, index_col=0)
    both = int(ct.loc["SD significant", "ID significant"])
    only_sd = int(ct.loc["SD significant", "ID not significant"])
    only_id = int(ct.loc["SD not significant", "ID significant"])
    neither = int(ct.loc["SD not significant", "ID not significant"])
    return {
        "both": both, "only_sd": only_sd, "only_id": only_id, "neither": neither,
        "n_at_least_one": both + only_sd + only_id,
        "n_sd_sig": both + only_sd, "n_id_sig": both + only_id,
        "net_gain_id": only_id - only_sd,
    }


def add_missing_note(doc, text):
    p = doc.add_paragraph(f"[MISSING: {text}]")
    p.runs[0].italic = True


def is_sig(p):
    """True only if p is available and below config.ALPHA."""
    return bool(pd.notna(p) and p < config.ALPHA)


def fmt_p(p):
    if pd.isna(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def chance_sentence(sd_ch, id_ch):
    """Sentence on classification vs. chance; wording depends on the actual p-values."""
    sd_sig = is_sig(sd_ch['p_value_vs_chance'])
    id_sig = is_sig(id_ch['p_value_vs_chance'])
    if sd_sig and id_sig:
        lead = "Both configurations classified motor states significantly above empirical chance level"
    elif id_sig:
        lead = ("Only the ID configuration classified motor states significantly above empirical "
                "chance level (the SD configuration did not)")
    elif sd_sig:
        lead = ("Only the SD configuration classified motor states significantly above empirical "
                "chance level (the ID configuration did not)")
    else:
        lead = "Neither configuration classified motor states significantly above empirical chance level"
    return (
        f"{lead}, as estimated via the restricted permutation test "
        f"(SD: p = {sd_ch['p_value_vs_chance']:.3f}, ID: p = {id_ch['p_value_vs_chance']:.3f}), "
        f"with effect sizes of Cohen's d = {sd_ch['cohens_d_vs_chance']:.2f} and "
        f"{id_ch['cohens_d_vs_chance']:.2f}, respectively."
    )

# ------------------------------------------------------------------
# Full group- and participant-level section (main analysis and appendix)
# ------------------------------------------------------------------

def build_full_section(doc, mode, heading, heading_level=1):
    paths = config.get_mode_paths(mode)
    output_dir = paths["output_dir"]
    prefix = paths["file_prefix"]

    doc.add_heading(heading, level=heading_level)

    # ---- Group level (ID vs. SD and vs. chance) ----
    doc.add_heading("Group-Level Analysis", level=heading_level + 1)

    group_stats_path = os.path.join(output_dir, f"{prefix}_group_stats.csv")
    chance_stats_path = os.path.join(output_dir, f"{prefix}_group_vs_chance_stats.csv")
    if not (os.path.exists(group_stats_path) and os.path.exists(chance_stats_path)):
        add_missing_note(doc, f"{prefix}_group_stats.csv / {prefix}_group_vs_chance_stats.csv "
                               f"(run 03_stats_report.py and 04_group_vs_chance.py for mode '{mode}')")
        return output_dir, prefix

    group_stats = pd.read_csv(group_stats_path).iloc[0]
    chance_stats = pd.read_csv(chance_stats_path).set_index("condition")
    sd_ch, id_ch = chance_stats.loc["SD"], chance_stats.loc["ID"]

    doc.add_paragraph(chance_sentence(sd_ch, id_ch))
    mdiff = group_stats['mean_diff_id_minus_sd']
    higher = "higher" if mdiff > 0 else "lower"
    if mdiff > 0:
        sig_part = ("This difference was statistically significant" if is_sig(group_stats['p_mean_diff'])
                    else "This difference was not statistically significant")
    else:
        sig_part = "This difference was not significant in the tested direction (ID > SD)"
    doc.add_paragraph(
        f"At the group level, mean classification accuracy was {higher} under the ID "
        f"configuration (M = {group_stats['mean_acc_id']*100:.1f}%, "
        f"SD = {group_stats['std_acc_id']*100:.1f}%) than under the SD configuration "
        f"(M = {group_stats['mean_acc_sd']*100:.1f}%, SD = {group_stats['std_acc_sd']*100:.1f}%), "
        f"a mean difference of {mdiff*100:.1f}%. {sig_part} (one-sided paired permutation test, "
        f"p = {group_stats['p_mean_diff']:.4f}), with Cohen's d = {group_stats['cohens_d']:.2f} "
        f"({group_stats['effect_size_words']})."
    )

    # ---- Participant level (ID vs. SD and vs. chance) ----
    doc.add_heading("Participant-Level Analysis", level=heading_level + 1)
    acc_stats_path = os.path.join(output_dir, f"{prefix}_accuracy_stats.csv")
    if not os.path.exists(acc_stats_path):
        add_missing_note(doc, f"{prefix}_accuracy_stats.csv (run 03_stats_report.py)")
        return output_dir, prefix

    acc_stats = pd.read_csv(acc_stats_path)
    n = len(acc_stats)
    diffs = acc_stats["diff_id_minus_sd"]
    n_id_higher = int((diffs > 0).sum())
    n_sd_higher = int((diffs < 0).sum())
    n_equal = n - n_id_higher - n_sd_higher
    q25, q75 = diffs.quantile(0.25), diffs.quantile(0.75)
    n_gain10 = int((diffs >= 0.10).sum())
    n_gain20 = int((diffs >= 0.20).sum())
    n_loss10 = int((diffs <= -0.10).sum())
    n_loss20 = int((diffs <= -0.20).sum())

    doc.add_paragraph(
        f"Numerically, {n_id_higher} of {n} participants ({100*n_id_higher/n:.1f}%) showed "
        f"higher accuracy under ID than under SD, compared to {n_sd_higher} "
        f"({100*n_sd_higher/n:.1f}%) showing the reverse pattern and {n_equal} "
        f"({100*n_equal/n:.1f}%) with no change (see Figure X). Accuracy differences ranged from "
        f"{diffs.min()*100:+.1f} to {diffs.max()*100:+.1f}% (median = {diffs.median()*100:+.1f}%, "
        f"IQR = {q25*100:+.1f} to {q75*100:+.1f}%). Gains of at least 10% were observed in "
        f"{100*n_gain10/n:.1f}% of participants (n = {n_gain10}), including "
        f"{100*n_gain20/n:.1f}% (n = {n_gain20}) with gains of at least 20%; losses of at least 10% "
        f"occurred in {100*n_loss10/n:.1f}% (n = {n_loss10}), including "
        f"{100*n_loss20/n:.1f}% (n = {n_loss20}) with losses of at least 20%."
    )

    sd_ind = load_condition_summary(output_dir, prefix, "SD")
    id_ind = load_condition_summary(output_dir, prefix, "ID")
    overlap = load_overlap(output_dir, prefix)
    if sd_ind is not None and id_ind is not None:
        n_sd_sig = int(sd_ind["significant_fdr"].sum())
        n_id_sig = int(id_ind["significant_fdr"].sum())
        if n_id_sig > n_sd_sig:
            cmp_txt = "More participants classified significantly above chance under ID"
        elif n_id_sig < n_sd_sig:
            cmp_txt = "Fewer participants classified significantly above chance under ID"
        else:
            cmp_txt = "The same number of participants classified significantly above chance under ID"
        doc.add_paragraph(
            f"{cmp_txt} "
            f"({n_id_sig}/{n}, {100*n_id_sig/n:.1f}%) as under SD "
            f"({n_sd_sig}/{n}, {100*n_sd_sig/n:.1f}%) (restricted permutation tests, FDR-corrected)."
            if n_id_sig == n_sd_sig else
            f"{cmp_txt} "
            f"({n_id_sig}/{n}, {100*n_id_sig/n:.1f}%) than under SD "
            f"({n_sd_sig}/{n}, {100*n_sd_sig/n:.1f}%) (restricted permutation tests, FDR-corrected)."
        )
    else:
        add_missing_note(doc, f"{prefix}_sd/id_individual_vs_chance.csv (run 04_group_vs_chance.py)")

    if overlap is not None:
        only_one = overlap['only_sd'] + overlap['only_id']
        if overlap['net_gain_id'] > 0:
            net_txt = f"representing a net gain of {overlap['net_gain_id']} participants under ID."
        elif overlap['net_gain_id'] < 0:
            net_txt = f"representing a net loss of {-overlap['net_gain_id']} participants under ID."
        else:
            net_txt = "representing no net difference between configurations."
        both_txt = ("Most " if overlap['both'] > overlap['n_at_least_one'] / 2 else "Of these, ")
        doc.add_paragraph(
            f"A total of {overlap['n_at_least_one']} participants reached significance in at "
            f"least one configuration. {both_txt}({overlap['both']}) did so under both configurations; "
            f"of the remaining {only_one} who reached significance "
            f"only under one, {overlap['only_id']} did so under ID and {overlap['only_sd']} under SD, "
            f"{net_txt}"
        )
    else:
        add_missing_note(doc, f"{prefix}_sd_id_overlap.csv (run 04_group_vs_chance.py)")

    n_sig_fdr = int(acc_stats["paired_p_id_vs_sd_sig_fdr"].sum())
    n_sig_raw = int(acc_stats["paired_p_id_vs_sd_sig_raw"].sum())
    if n_sig_raw == 0:
        fdr_txt = "No participant showed a significant SD-ID difference, neither uncorrected nor after FDR correction."
    else:
        fdr_txt = (f"After FDR correction, {n_sig_fdr} participant(s) showed a significant SD-ID difference"
                   f"{' (none survived FDR correction)' if n_sig_fdr == 0 else ''}.")
    doc.add_paragraph(
        f"The ID accuracy increase reached uncorrected significance for {n_sig_raw} "
        f"of {n} participants ({100*n_sig_raw/n:.1f}%). " + fdr_txt
    )

    fig_path = os.path.join(output_dir, "accuracy_change_sorted_uncorrected.png")
    if os.path.exists(fig_path):
        doc.add_picture(fig_path, width=Inches(6.5))
        cap = doc.add_paragraph(
            "Figure X. Per-participant accuracy change (ID minus SD), not corrected for multiple comparisons."
        )
        cap.runs[0].italic = True
    else:
        add_missing_note(doc, "accuracy_change_sorted_uncorrected.png (run 03_stats_report.py)")

    return output_dir, prefix


# ------------------------------------------------------------------
# Data-quality sensitivity analysis
# ------------------------------------------------------------------

def build_quality_section(doc):
    output_dir = config.RESULTS_QUALITY_DIR
    doc.add_heading("Data-Quality Sensitivity Analysis", level=2)

    quality_files = sorted(glob.glob(os.path.join(output_dir, "quality_overview_*.xlsx")))
    if not quality_files:
        add_missing_note(doc, "quality_overview_*.xlsx (run 06_quality_channel_labels.py)")
        return
    quality_df = pd.read_excel(quality_files[-1])
    counts = quality_df.drop_duplicates("Participant")["Combined_Quality_Label"].value_counts()
    n_total = int(counts.sum())

    group_stats_path = os.path.join(output_dir, "accuracy_by_quality_group_stats.csv")
    if not os.path.exists(group_stats_path):
        add_missing_note(doc, "accuracy_by_quality_group_stats.csv (run 07_quality_group_stats.py)")
        return
    gdf = pd.read_csv(group_stats_path).set_index("quality_label")

    labels = ["Good", "Moderate", "Poor"]
    parts = [f"{int(counts.get(l, 0))} ({100*counts.get(l, 0)/n_total:.1f}%) {l}" for l in labels]
    doc.add_paragraph(f"Among the {n_total} datasets, " + ", ".join(parts) + " quality.")

    valid_diffs, valid_ds = [], []
    for label in labels:
        if label not in gdf.index or pd.isna(gdf.loc[label, "p_value"]):
            continue
        row = gdf.loc[label]
        p_h = row["p_value_holm"]
        sig_note = "significant" if is_sig(p_h) else "not significant"
        doc.add_paragraph(
            f"In the {label} quality group, mean accuracy was {row['mean_acc_sd']*100:.1f}% "
            f"(\u00b1 {row['std_acc_sd']*100:.1f}%) under SD and {row['mean_acc_id']*100:.1f}% "
            f"(\u00b1 {row['std_acc_id']*100:.1f}%) under ID (mean difference ID \u2212 SD = "
            f"{row['mean_diff_id_minus_sd']*100:.1f}%, p = {row['p_value']:.3f}, "
            f"Holm-corrected p = {p_h:.3f} [{sig_note} at \u03b1 = {config.ALPHA}], "
            f"Cohen's d = {row['cohens_d']:.3f}, {row['effect_size_words']} effect)."
        )
        valid_diffs.append(row["mean_diff_id_minus_sd"])
        valid_ds.append(row["cohens_d"])

    if len(valid_diffs) >= 2:
        if all(d > 0 for d in valid_diffs):
            lead = "Accuracy was higher under ID than under SD in all quality groups"
        elif all(d < 0 for d in valid_diffs):
            lead = "Accuracy was lower under ID than under SD in all quality groups"
        else:
            lead = "The direction of the ID-SD difference was not consistent across quality groups"
        order = ["negligible", "small", "medium", "large"]
        words = sorted({effect_size_words(d) for d in valid_ds},
                       key=lambda w: order.index(w) if w in order else 99)
        effect_txt = words[0] if len(words) == 1 else f"{words[0]} to {words[-1]}"
        doc.add_paragraph(
            f"{lead} (mean differences ranging from {min(valid_diffs)*100:.1f}% to "
            f"{max(valid_diffs)*100:.1f}%; effect sizes {effect_txt}, "
            f"d = {min(valid_ds):.2f}\u2013{max(valid_ds):.2f}; "
            f"p-values were Holm-corrected for the {len(valid_diffs)} group-wise tests).")


# ------------------------------------------------------------------
# Shared latent space sensitivity analysis (condensed version for the main text)
# ------------------------------------------------------------------

def build_sensitivity_condensed(doc):
    doc.add_heading("Shared Latent Space Sensitivity Analysis", level=2)

    paths = config.get_mode_paths("sensitivity")
    output_dir, prefix = paths["output_dir"], paths["file_prefix"]

    group_stats_path = os.path.join(output_dir, f"{prefix}_group_stats.csv")
    acc_stats_path = os.path.join(output_dir, f"{prefix}_accuracy_stats.csv")
    if not (os.path.exists(group_stats_path) and os.path.exists(acc_stats_path)):
        add_missing_note(doc, "sensitivity outputs (run steps 02-04 with --mode sensitivity)")
        return

    group_stats = pd.read_csv(group_stats_path).iloc[0]
    acc_stats = pd.read_csv(acc_stats_path)
    n_id_higher = int((acc_stats["diff_id_minus_sd"] > 0).sum())
    n_sd_higher = int((acc_stats["diff_id_minus_sd"] < 0).sum())
    sd_ind = load_condition_summary(output_dir, prefix, "SD")
    id_ind = load_condition_summary(output_dir, prefix, "ID")

    doc.add_paragraph(
        "To rule out that the SD-vs-ID advantage was driven by the difference in feature "
        "dimensionality between the two configurations (13 vs. 20 channels), both conditions "
        "were additionally projected into an identical low-dimensional latent space producing "
        "a fixed six-dimensional representation for both SD and ID."
    )
    sens_diff = group_stats['mean_diff_id_minus_sd']
    sens_p = group_stats['p_mean_diff']
    sens_ok = bool(sens_diff > 0 and is_sig(sens_p))

    chance_path = os.path.join(output_dir, f"{prefix}_group_vs_chance_stats.csv")
    chance_txt = ""
    if os.path.exists(chance_path):
        cs = pd.read_csv(chance_path).set_index("condition")
        sd_sig = is_sig(cs.loc["SD", "p_value_vs_chance"])
        id_sig = is_sig(cs.loc["ID", "p_value_vs_chance"])
        if sd_sig and id_sig:
            chance_txt = "both configurations classified motor states significantly above chance level, and "
        elif id_sig:
            chance_txt = "only the ID configuration classified motor states significantly above chance level, and "
        elif sd_sig:
            chance_txt = "only the SD configuration classified motor states significantly above chance level, and "
        else:
            chance_txt = "neither configuration classified motor states significantly above chance level, and "

    higher_word = "higher" if sens_diff > 0 else "lower"
    if sens_diff > 0:
        sig_sentence = ("a statistically significant" if is_sig(sens_p)
                        else "a statistically non-significant")
    else:
        sig_sentence = "a (non-significant in the tested ID > SD direction)"
    doc.add_paragraph(
        f"At the group level, in this dimensionality-controlled comparison {chance_txt}"
        f"mean classification accuracy was {higher_word} under ID (M = {group_stats['mean_acc_id']*100:.1f}%, "
        f"SD = {group_stats['std_acc_id']*100:.1f}%) than under SD (M = {group_stats['mean_acc_sd']*100:.1f}%, "
        f"SD = {group_stats['std_acc_sd']*100:.1f}%), {sig_sentence} mean difference "
        f"of {sens_diff*100:.1f}% ({fmt_p(sens_p)})."
    )

    # Individual level is reported without counts here (direction only); exact counts are in the appendix.
    sig_txt = ""
    if sd_ind is not None and id_ind is not None:
        n_sd_sig = int(sd_ind["significant_fdr"].sum())
        n_id_sig = int(id_ind["significant_fdr"].sum())
        if n_id_sig > n_sd_sig:
            sig_txt = ", and more participants classified significantly above chance under ID than under SD"
        elif n_id_sig < n_sd_sig:
            sig_txt = ", and fewer participants classified significantly above chance under ID than under SD"
        else:
            sig_txt = ", and the same number of participants classified significantly above chance under ID and SD"

    if n_id_higher > n_sd_higher:
        direction_txt = "more participants showed numerically higher accuracy under ID than under SD"
    elif n_id_higher < n_sd_higher:
        direction_txt = "fewer participants showed numerically higher accuracy under ID than under SD"
    else:
        direction_txt = "equally many participants showed numerically higher accuracy under ID and under SD"

    doc.add_paragraph(
        f"At the individual level, {direction_txt}{sig_txt}. "
        f"Full participant-level statistics, including the SD-ID overlap, are reported in the Appendix."
    )

    # Compare with the main analysis using the actual values only
    main_paths = config.get_mode_paths("main")
    main_gs_path = os.path.join(main_paths["output_dir"], f"{main_paths['file_prefix']}_group_stats.csv")
    main_ok = None
    if os.path.exists(main_gs_path):
        mg = pd.read_csv(main_gs_path).iloc[0]
        main_ok = bool(mg['mean_diff_id_minus_sd'] > 0 and is_sig(mg['p_mean_diff']))

    if sens_ok and main_ok is not False:
        closing = ("Overall, the dimensionality-controlled analysis reproduced the direction and statistical "
                   "significance of the main finding, suggesting that the ID advantage was not simply "
                   "an artifact of ID's higher native channel count.")
    elif sens_diff > 0:
        closing = ("Overall, the dimensionality-controlled analysis showed a numerically higher accuracy "
                   "under ID, but this difference did not reach statistical significance; the analysis "
                   "therefore does not provide clear evidence on whether the ID advantage depends on "
                   "ID's higher native channel count.")
    else:
        closing = ("Overall, the dimensionality-controlled analysis did not reproduce an accuracy "
                   "advantage of ID over SD.")
    doc.add_paragraph(closing)


# ------------------------------------------------------------------
# Appendix: methodological robustness checks (pipeline choice, pipeline stability, trial count)
# ------------------------------------------------------------------

def build_robustness_section(doc, mode, heading, heading_level=1):
    paths = config.get_mode_paths(mode)
    output_dir, prefix = paths["output_dir"], paths["file_prefix"]
    doc.add_heading(heading, level=heading_level)

    # ---- Pipeline choice SD vs. ID (chi-square) ----
    chi2_path = os.path.join(output_dir, f"{prefix}_appendix_pipeline_chi2.csv")
    if os.path.exists(chi2_path):
        row = pd.read_csv(chi2_path).iloc[0]
        if row["top_pipeline_sd"] == row["top_pipeline_id"]:
            same_txt = f"The most frequently selected pipeline was the same for both SD and ID ({row['top_pipeline_sd']})."
        else:
            same_txt = (f"The most frequently selected pipeline differed between SD "
                        f"({row['top_pipeline_sd']}) and ID ({row['top_pipeline_id']}).")
        sig_txt = "differed significantly" if row["p_value"] < config.ALPHA else "did not differ significantly"
        doc.add_paragraph(
            f"{same_txt} A chi-square test indicated that overall pipeline-selection frequency "
            f"{sig_txt} between conditions (\u03c7\u00b2({int(row['dof'])}) = {row['chi2']:.2f}, "
            f"p = {row['p_value']:.3f})."
        )
    else:
        add_missing_note(doc, f"{prefix}_appendix_pipeline_chi2.csv "
                               f"(re-run 03_stats_report.py)")

    # ---- Pipeline stability and trial count ----
    appendix_path = os.path.join(output_dir, f"{prefix}_appendix_stability_trialcount.csv")
    if os.path.exists(appendix_path):
        df = pd.read_csv(appendix_path)
        checks = [
            ("pipeline_stability_vs_fold_variance", "Pipeline stability vs. fold-accuracy variance",
             "whether unstable pipeline selection across folds was associated with less stable accuracy"),
            ("trial_count_vs_accuracy", "Trial count vs. accuracy",
             "whether the number of available trials was associated with classification accuracy"),
        ]
        for analysis, label, desc in checks:
            sub = df[df["analysis"] == analysis]
            if sub.empty:
                continue
            parts = []
            for _, r in sub.iterrows():
                if pd.isna(r["spearman_r"]):
                    parts.append(f"{r['condition']}: insufficient data (n={int(r['n'])})")
                else:
                    parts.append(f"{r['condition']}: Spearman r = {r['spearman_r']:.3f}, "
                                  f"p = {r['p_value']:.3f} (n={int(r['n'])})")
            doc.add_paragraph(f"{label} (checks {desc}): " + "; ".join(parts) + ".")
    else:
        add_missing_note(doc, f"{prefix}_appendix_stability_trialcount.csv "
                               f"(re-run 03_stats_report.py)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    doc = Document()
    doc.add_heading("Results", level=0)

    build_full_section(doc, "main", "Multivariate Results", heading_level=1)
    build_quality_section(doc)
    build_sensitivity_condensed(doc)

    doc.add_page_break()
    doc.add_heading("Appendix", level=0)
    build_robustness_section(
        doc, "main",
        "Appendix A: Methodological Robustness Checks (Main Analysis)",
        heading_level=1,
    )
    build_full_section(
        doc, "sensitivity",
        "Appendix B: Shared Latent Space Sensitivity Analysis - Full Statistics",
        heading_level=1,
    )
    build_robustness_section(
        doc, "sensitivity",
        "Appendix C: Methodological Robustness Checks (Sensitivity Analysis)",
        heading_level=1,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = os.path.join(config.CUSTOM_OUTPUT_ROOT, f"results_summary_{timestamp}.docx")
    doc.save(out_path)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
