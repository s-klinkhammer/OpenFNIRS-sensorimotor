"""
07_quality_group_stats.py
Splits the main-analysis nested-CV results by signal quality (Good / Moderate /
Poor) and computes per group:
  - mean / SD accuracy for SD and ID
  - group-level permutation test ID vs. SD (same method as the group-level comparison in
    03_stats_report.py, applied to the subgroup)
  - effect size (Cohen's d + verbal label)
p-values are Holm-corrected across the three quality groups.

Inputs : quality_overview_*.xlsx from 06_quality_channel_labels.py and the
         latest main-analysis summary / trial-prediction CSVs.
Usage  : python 07_quality_group_stats.py [--quality-xlsx FILE]
"""

import os
import re
import argparse

import numpy as np
import pandas as pd

import config
from core.stats_core import group_level_permutation_test, cohens_d_paired, effect_size_words, fdr_correct

parser = argparse.ArgumentParser(description="Statistics per signal-quality group")
parser.add_argument("--quality-xlsx", help="quality_overview_*.xlsx from step 06 (default: latest)")
args = parser.parse_args()

main_paths = config.get_mode_paths("main")
output_dir = config.RESULTS_QUALITY_DIR

summary_file = config.find_latest(main_paths["output_dir"], f"{main_paths['file_prefix']}_summary")
trials_file = config.find_latest(main_paths["output_dir"], f"{main_paths['file_prefix']}_trial_predictions")
quality_path = args.quality_xlsx or config.find_latest(config.RESULTS_QUALITY_DIR, "quality_overview_", ".xlsx")
print(f"Summary: {summary_file}\nTrials:  {trials_file}\nQuality: {quality_path}")

summary_df = pd.read_csv(summary_file)
trials_df = pd.read_csv(trials_file)
quality_df = pd.read_excel(quality_path)[["Participant", "Combined_Quality_Label"]]


def _pid(x):
    """Normalise participant IDs ('sub-001', 'P001', '1') to a digit string for matching."""
    return re.sub(r"\D", "", str(x)).lstrip("0")


summary_df["_pid"] = summary_df["participant"].map(_pid)
trials_df["_pid"] = trials_df["participant"].map(_pid)
quality_df["_pid"] = quality_df["Participant"].map(_pid)

rng = np.random.default_rng(seed=config.RANDOM_STATE)
group_stat_rows = []

for label in ["Good", "Moderate", "Poor"]:
    label_pids = quality_df.loc[quality_df["Combined_Quality_Label"] == label, "_pid"]
    summary_out = summary_df[summary_df["_pid"].isin(label_pids)]
    trials_out = trials_df[trials_df["_pid"].isin(label_pids)]
    summary_out.to_csv(os.path.join(output_dir, f"participant_summary_{label.lower()}.csv"), index=False)
    trials_out.to_csv(os.path.join(output_dir, f"participant_trial_predictions_{label.lower()}.csv"), index=False)

    # Per-participant accuracy for this group (only participants with both conditions)
    acc_rows = []
    for p in sorted(trials_out["participant"].unique()):
        p_trials = trials_out[trials_out["participant"] == p]
        row, complete = {"participant": p}, True
        for cond in config.CONDITIONS:
            cond_trials = p_trials[p_trials["condition"] == cond]
            if cond_trials.empty:
                complete = False
                continue
            row[f"acc_{cond.lower()}"] = (cond_trials["y_true"] == cond_trials["y_pred"]).mean()
        if complete:
            acc_rows.append(row)
    df_group = pd.DataFrame(acc_rows)

    if len(df_group) < 2:
        group_stat_rows.append({
            "quality_label": label, "n_participants": len(df_group),
            "mean_acc_sd": np.nan, "std_acc_sd": np.nan,
            "mean_acc_id": np.nan, "std_acc_id": np.nan,
            "mean_diff_id_minus_sd": np.nan, "p_value": np.nan,
            "cohens_d": np.nan, "effect_size_words": "n/a",
        })
        continue

    obs_diff, p_value = group_level_permutation_test(
        df_group["acc_sd"].values, df_group["acc_id"].values, config.N_PERM, rng)
    d = cohens_d_paired((df_group["acc_id"] - df_group["acc_sd"]).values)

    group_stat_rows.append({
        "quality_label": label, "n_participants": len(df_group),
        "mean_acc_sd": df_group["acc_sd"].mean(), "std_acc_sd": df_group["acc_sd"].std(),
        "mean_acc_id": df_group["acc_id"].mean(), "std_acc_id": df_group["acc_id"].std(),
        "mean_diff_id_minus_sd": obs_diff, "p_value": p_value,
        "cohens_d": d, "effect_size_words": effect_size_words(d),
    })

df_group_stats = pd.DataFrame(group_stat_rows)

# Holm correction across the quality groups
valid = df_group_stats["p_value"].notna()
df_group_stats["p_value_holm"] = np.nan
df_group_stats["significant_holm"] = np.nan
if valid.any():
    rej, p_holm = fdr_correct(df_group_stats.loc[valid, "p_value"].values,
                              alpha=config.ALPHA, method="holm")
    df_group_stats.loc[valid, "p_value_holm"] = p_holm
    df_group_stats.loc[valid, "significant_holm"] = rej

group_stats_file = os.path.join(output_dir, "accuracy_by_quality_group_stats.csv")
df_group_stats.round(3).to_csv(group_stats_file, index=False)

print(df_group_stats.to_string(index=False))
print(f"\nSaved: {group_stats_file}")
