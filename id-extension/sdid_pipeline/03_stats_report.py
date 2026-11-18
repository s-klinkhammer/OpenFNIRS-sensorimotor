"""
03_stats_report.py
Statistics report for the main or sensitivity analysis.

Usage: python 03_stats_report.py --mode main|sensitivity

Computes
  Group level       ID vs. SD (mean/SD, permutation test, effect sizes)
  Individual level  ID-vs-SD differences per participant (paired permutation test, FDR) + plots
  Appendix          pipeline distribution SD vs. ID (chi-square), pipeline stability vs.
                    fold variance, trial count vs. accuracy

The comparisons against chance (group and individual level) are intentionally
NOT computed here but in 04_group_vs_chance.py, because they require a
restricted permutation test in which the pipeline is refit per permutation.
The SD-vs-ID comparisons here compare two finished results with each other,
for which the cheaper post-hoc permutation test (label swap) is appropriate.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr

import config
from core.stats_core import (
    rank_biserial, cohens_d_paired, effect_size_words,
    paired_permutation_test, group_level_permutation_test,
    fdr_correct,
)


MODE = config.resolve_mode()
INCLUDE_APPENDIX_CHECKS = True   # also compute the appendix analyses (pipeline stability, trial count)

paths = config.get_mode_paths(MODE)
output_dir = paths["output_dir"]
summary_file = config.find_latest(output_dir, f"{paths['file_prefix']}_summary")
trials_file = config.find_latest(output_dir, f"{paths['file_prefix']}_trial_predictions")

print(f"[{MODE}] Summary: {summary_file}")
print(f"[{MODE}] Trials:  {trials_file}")

df_summary = pd.read_csv(summary_file)
df_trials = pd.read_csv(trials_file)

participants = sorted(df_summary['participant'].unique())
conditions = list(config.CONDITIONS)
n_total = len(participants)
rng = np.random.default_rng(seed=config.RANDOM_STATE)

# ---- Part 1: per-participant accuracy (SD/ID) + paired comparison (individual level) ----
rows = []
for p in participants:
    p_trials = df_trials[df_trials['participant'] == p]
    res = {'participant': p}

    for cond in conditions:
        cond_trials = p_trials[p_trials['condition'] == cond]
        res[f'acc_{cond.lower()}'] = (cond_trials['y_true'] == cond_trials['y_pred']).mean()

    sd_trials = p_trials[p_trials['condition'] == 'SD']
    id_trials = p_trials[p_trials['condition'] == 'ID']
    sd_corr = (sd_trials['y_true'] == sd_trials['y_pred']).astype(int).values
    id_corr = (id_trials['y_true'] == id_trials['y_pred']).astype(int).values

    obs_diff, p_paired = paired_permutation_test(sd_corr, id_corr, config.N_PERM, rng)
    res['paired_p_id_vs_sd'] = p_paired
    res['diff_id_minus_sd'] = obs_diff

    rows.append(res)

df_participant_stats = pd.DataFrame(rows)

reject, p_fdr = fdr_correct(df_participant_stats['paired_p_id_vs_sd'].values, alpha=config.ALPHA, method=config.FDR_METHOD)
df_participant_stats['paired_p_id_vs_sd_fdr'] = p_fdr
df_participant_stats['paired_p_id_vs_sd_sig_fdr'] = reject
df_participant_stats['paired_p_id_vs_sd_sig_raw'] = df_participant_stats['paired_p_id_vs_sd'] < config.ALPHA

# ---- Part 2: group-level mean/SD + permutation test ID vs. SD (group level) ----
mean_sd, std_sd = df_participant_stats['acc_sd'].mean(), df_participant_stats['acc_sd'].std()
mean_id, std_id = df_participant_stats['acc_id'].mean(), df_participant_stats['acc_id'].std()

obs_mean_diff, p_mean_diff = group_level_permutation_test(
    df_participant_stats['acc_sd'].values, df_participant_stats['acc_id'].values,
    config.N_PERM, rng
)

participant_diffs = (df_participant_stats['acc_id'] - df_participant_stats['acc_sd']).values
d_cohen = cohens_d_paired(participant_diffs)
rrb_group = rank_biserial(participant_diffs)
prop_id_greater = float((participant_diffs > 0).mean())

df_group_stats = pd.DataFrame([{
    'mean_acc_sd': mean_sd, 'std_acc_sd': std_sd,
    'mean_acc_id': mean_id, 'std_acc_id': std_id,
    'mean_diff_id_minus_sd': obs_mean_diff, 'p_mean_diff': p_mean_diff,
    'cohens_d': d_cohen, 'effect_size_words': effect_size_words(d_cohen),
    'rank_biserial_r': rrb_group,
    'prop_participants_id_greater_sd': prop_id_greater,
}])

# ---- Part 3 [appendix]: does the chosen pipeline differ between SD and ID? ----
pipeline_records = []
for _, row in df_summary.iterrows():
    for f in json.loads(row['chosen_models']):
        pipeline_records.append({
            'participant': row['participant'], 'condition': row['condition'],
            'fold': f['fold'], 'pipeline': f['pipeline'],
            'fold_accuracy': f.get('fold_accuracy', np.nan)
        })
df_pipelines = pd.DataFrame(pipeline_records)

appendix_pipeline = None
if not df_pipelines.empty and df_pipelines['pipeline'].notna().any():
    contingency = pd.crosstab(df_pipelines['pipeline'], df_pipelines['condition'])
    chi2, p_chi2, dof, _ = chi2_contingency(contingency)
    top_by_cond = df_pipelines.groupby('condition')['pipeline'].agg(lambda s: s.value_counts().idxmax())
    appendix_pipeline = {
        'chi2': chi2, 'dof': dof, 'p_value': p_chi2,
        'top_pipeline_sd': top_by_cond.get('SD', 'n/a'),
        'top_pipeline_id': top_by_cond.get('ID', 'n/a'),
    }
    contingency.to_csv(os.path.join(output_dir, f'{paths["file_prefix"]}_pipeline_distribution.csv'))
    pd.DataFrame([{
        'chi2': chi2, 'dof': dof, 'p_value': p_chi2,
        'top_pipeline_sd': appendix_pipeline['top_pipeline_sd'],
        'top_pipeline_id': appendix_pipeline['top_pipeline_id'],
    }]).to_csv(os.path.join(output_dir, f'{paths["file_prefix"]}_appendix_pipeline_chi2.csv'), index=False)

# ---- Part 4 [appendix, optional]: pipeline stability vs. fold variance,
#      trial count vs. accuracy ----
stability_result, trial_count_result = {}, {}
if INCLUDE_APPENDIX_CHECKS:
    if not df_pipelines.empty and df_pipelines['fold_accuracy'].notna().any():
        stab_rows = []
        for (p, cond), grp in df_pipelines.groupby(['participant', 'condition']):
            modal = grp['pipeline'].value_counts().idxmax()
            stability = (grp['pipeline'] == modal).mean()
            fold_acc_std = grp['fold_accuracy'].std(ddof=1) if len(grp) > 1 else np.nan
            stab_rows.append({'participant': p, 'condition': cond, 'n_folds': len(grp),
                               'modal_pipeline': modal, 'pipeline_stability': stability,
                               'fold_accuracy_std': fold_acc_std})
        df_stability = pd.DataFrame(stab_rows)
        df_stability.to_csv(os.path.join(output_dir, f'{paths["file_prefix"]}_pipeline_stability.csv'), index=False)

        for cond in conditions:
            sub = df_stability[(df_stability['condition'] == cond) & df_stability['fold_accuracy_std'].notna()]
            if len(sub) >= 3:
                rho, p_rho = spearmanr(sub['pipeline_stability'], sub['fold_accuracy_std'])
            else:
                rho, p_rho = np.nan, np.nan
            stability_result[cond] = {'n': len(sub), 'spearman_r': rho, 'p_value': p_rho}

    if 'n_trials' in df_summary.columns:
        trial_counts = df_summary.pivot(index='participant', columns='condition', values='n_trials')
        merged = df_participant_stats.set_index('participant').join(
            trial_counts.rename(columns={'SD': 'n_trials_sd', 'ID': 'n_trials_id'})
        )
        for cond in conditions:
            n_col, acc_col = f'n_trials_{cond.lower()}', f'acc_{cond.lower()}'
            sub = merged[[n_col, acc_col]].dropna()
            if len(sub) >= 3:
                rho, p_rho = spearmanr(sub[n_col], sub[acc_col])
            else:
                rho, p_rho = np.nan, np.nan
            trial_count_result[cond] = {'n': len(sub), 'spearman_r': rho, 'p_value': p_rho}

    # Save the appendix analyses as CSV (read by 08_build_results_document.py)
    appendix_rows = []
    for cond, val in stability_result.items():
        appendix_rows.append({'analysis': 'pipeline_stability_vs_fold_variance', 'condition': cond, **val})
    for cond, val in trial_count_result.items():
        appendix_rows.append({'analysis': 'trial_count_vs_accuracy', 'condition': cond, **val})
    if appendix_rows:
        pd.DataFrame(appendix_rows).to_csv(
            os.path.join(output_dir, f'{paths["file_prefix"]}_appendix_stability_trialcount.csv'), index=False
        )

# ---- Save core statistics tables ----
part_stats_file = os.path.join(output_dir, f'{paths["file_prefix"]}_accuracy_stats.csv')
group_stats_file = os.path.join(output_dir, f'{paths["file_prefix"]}_group_stats.csv')
df_participant_stats.to_csv(part_stats_file, index=False)
df_group_stats.to_csv(group_stats_file, index=False)

# ---- Plots: per-participant accuracy change, uncorrected / FDR-corrected ----
df_plot = df_participant_stats.sort_values(by='diff_id_minus_sd', ascending=False)


def plot_accuracy_change(df_plot, sig_col, title_suffix, out_path):
    plt.figure(figsize=(18, 6))
    ax = plt.gca()
    colors = ['#6AB187' if v > 0 else '#FA8C64' for v in df_plot['diff_id_minus_sd']]
    bars = ax.bar(df_plot['participant'], df_plot['diff_id_minus_sd'], color=colors)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_ylabel("Accuracy change (ID - SD)")
    ax.set_xlabel("Participant")
    ax.set_title(f"Per-participant accuracy change (ID - SD, sorted) - {title_suffix}")
    plt.xticks(rotation=90)
    for i, sig in enumerate(df_plot[sig_col]):
        if sig:
            bar = bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     "*", ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


plot_corrected = os.path.join(output_dir, "accuracy_change_sorted_corrected.png")
plot_accuracy_change(df_plot, 'paired_p_id_vs_sd_sig_fdr', '* = FDR-corrected significant', plot_corrected)

plot_uncorrected = os.path.join(output_dir, "accuracy_change_sorted_uncorrected.png")
plot_accuracy_change(df_plot, 'paired_p_id_vs_sd_sig_raw', '* = uncorrected significant', plot_uncorrected)

# ---- Consolidated text report ----
n_sig_paired_fdr = int(df_participant_stats['paired_p_id_vs_sd_sig_fdr'].sum())
n_id_higher = int((df_participant_stats['diff_id_minus_sd'] > 0).sum())

lines = []
lines.append("=" * 78)
lines.append(f"STATISTICS SUMMARY [{MODE.upper()}]")
lines.append("SD vs. ID Optode Configuration Classification")
lines.append("=" * 78)
lines.append(f"N participants analyzed: {n_total}")
lines.append(f"Significance threshold: alpha = {config.ALPHA}")
lines.append(f"Multiple-comparisons correction: Benjamini-Hochberg FDR ({config.FDR_METHOD})")
lines.append(f"Permutations per test: N = {config.N_PERM} (post-hoc label-shuffle test - see script docstring)")
lines.append("")

# Group level: ID vs. SD
lines.append("-" * 78)
lines.append("GROUP LEVEL (ID vs. SD): Does group-level classification accuracy differ significantly")
lines.append("between ID and SD, when each configuration's pipeline is optimized?")
lines.append("-" * 78)
gs = df_group_stats.iloc[0]
lines.append(f"Mean accuracy SD: {gs['mean_acc_sd']:.3f} (SD +/- {gs['std_acc_sd']:.3f})")
lines.append(f"Mean accuracy ID: {gs['mean_acc_id']:.3f} (SD +/- {gs['std_acc_id']:.3f})")
lines.append(f"Permutation test p-value: {gs['p_mean_diff']:.4f}")
lines.append(f"Effect size: d = {gs['cohens_d']:.3f} ({gs['effect_size_words']})")
lines.append(f"(non-parametric: rank-biserial r = {gs['rank_biserial_r']:.3f}; "
             f"{100*gs['prop_participants_id_greater_sd']:.0f}% of participants show ID > SD)")
verdict1 = (f"ID significantly outperforms SD at the group level (at alpha = {config.ALPHA})."
            if gs['p_mean_diff'] < config.ALPHA else
            f"No significant group-level difference between ID and SD (at alpha = {config.ALPHA}).")
lines.append(f"=> {verdict1}")
lines.append("")

# Group level: vs. chance (computed in 04)
lines.append("-" * 78)
lines.append("GROUP LEVEL (vs. chance): Does SD/ID classification exceed chance at the group level?")
lines.append("-" * 78)
lines.append("=> Answered in 04_group_vs_chance.py (restricted permutation test - "
             "pipeline is refit per permutation).")
lines.append("")

# Individual level: ID vs. SD
lines.append("-" * 78)
lines.append("INDIVIDUAL LEVEL (ID vs. SD): For how many individuals does accuracy differ significantly between")
lines.append("ID and SD (optimized pipelines)?")
lines.append("-" * 78)
lines.append(f"ID better than SD: {n_id_higher}/{n_total} ({100*n_id_higher/n_total:.1f}%)")
lines.append(f"ID significantly better than SD (FDR-corrected): {n_sig_paired_fdr}/{n_total} "
             f"({100*n_sig_paired_fdr/n_total:.1f}%)")
lines.append(f"=> See {os.path.basename(plot_uncorrected)} / {os.path.basename(plot_corrected)}")
lines.append("")

# Individual level: vs. chance (computed in 04)
lines.append("-" * 78)
lines.append("INDIVIDUAL LEVEL (vs. chance): Does SD/ID accuracy exceed chance at the individual level "
             "(+ overlap/asymmetry)?")
lines.append("-" * 78)
lines.append("=> Answered in 04_group_vs_chance.py (same restricted permutation "
             "refits as the group-level test, computed once and reused for both).")
lines.append("")

# Appendix: pipeline distribution
lines.append("-" * 78)
lines.append("APPENDIX: Does the optimal pipeline systematically differ between SD and ID?")
lines.append("-" * 78)
if appendix_pipeline is not None:
    if appendix_pipeline['top_pipeline_sd'] == appendix_pipeline['top_pipeline_id']:
        lines.append(f"The most frequently chosen pipeline for both SD and ID is: "
                     f"{appendix_pipeline['top_pipeline_sd']}")
    else:
        lines.append(f"Top pipeline SD: {appendix_pipeline['top_pipeline_sd']}, "
                     f"top pipeline ID: {appendix_pipeline['top_pipeline_id']}")
    lines.append(f"chi2({appendix_pipeline['dof']}) = {appendix_pipeline['chi2']:.2f}, "
                 f"p = {appendix_pipeline['p_value']:.3f}")
    verdict4 = ("The preferred pipeline differs significantly between SD and ID."
                if appendix_pipeline['p_value'] < config.ALPHA else
                "The preferred preprocessing/classification pipeline appears to be "
                "similar across the two conditions.")
    lines.append(f"=> {verdict4}")
else:
    lines.append("NOT AVAILABLE (no pipeline data found in chosen_models).")
lines.append("")

if INCLUDE_APPENDIX_CHECKS:
    lines.append("-" * 78)
    lines.append("APPENDIX: Does unstable pipeline selection across folds relate to")
    lines.append("unstable accuracy?")
    lines.append("-" * 78)
    for cond, val in stability_result.items():
        if np.isnan(val['spearman_r']):
            lines.append(f"{cond}: insufficient data (n={val['n']})")
        else:
            lines.append(f"{cond}: Spearman r = {val['spearman_r']:.3f}, p = {val['p_value']:.4f} (n={val['n']})")
    lines.append("")

    lines.append("-" * 78)
    lines.append("APPENDIX: Is accuracy associated with the number of available trials?")
    lines.append("-" * 78)
    for cond, val in trial_count_result.items():
        if np.isnan(val['spearman_r']):
            lines.append(f"{cond}: insufficient data (n={val['n']})")
        else:
            lines.append(f"{cond}: Spearman r = {val['spearman_r']:.3f}, p = {val['p_value']:.4f} (n={val['n']})")
    lines.append("")


lines.append("=" * 78)
lines.append("END OF REPORT")
lines.append("=" * 78)

report_text = "\n".join(lines)
report_file = os.path.join(output_dir, f"{paths['file_prefix']}_stats_summary.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n[{MODE}] Report saved: {report_file}")

# %%