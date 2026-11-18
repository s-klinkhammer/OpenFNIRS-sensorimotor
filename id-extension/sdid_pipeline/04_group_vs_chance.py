"""
04_group_vs_chance.py
Does classification accuracy in SD/ID exceed chance level at the group level
and at the individual level (including the overlap/asymmetry between SD and ID)?

Usage: python 04_group_vs_chance.py --mode main|sensitivity [--recompute]

Method: RESTRICTED PERMUTATION TEST (Ojala & Garriga, 2010). For every
permutation the pipeline is actually REFIT on the permuted labels (full
re-evaluation); only the hyperparameter grid search is omitted (default
hyperparameters). The feature-extraction method is fixed once per
participant and condition to the method chosen most often in the real
nested-CV analysis. The classifier is re-selected among all candidates for
the observed data and for every permutation (see core/cv_core.py:
refit_best_classifier_cv). Neither the classifier choice nor the predictions
are reused from the real analysis.

The group- and individual-level tests deliberately share the same refits.
The SD-vs-ID comparisons are in 03_stats_report.py and use the cheaper post-hoc test.

Runtime: n_perm + 1 fits per participant and condition (each with
outer_splits folds). Reduce config.N_PERM_RESTRICTED or run overnight if
needed. Permutation results are cached per condition; --recompute ignores
the cache.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from core.cv_core import (
    build_pipelines_main, build_pipelines_shared_space,
    modal_fe_method_per_participant, restricted_permutation_null_accuracies,
)
from core.stats_core import (
    restricted_group_p_value, restricted_individual_p_values,
    cohens_d_paired, rank_biserial, effect_size_words, fdr_correct,
)

RECOMPUTE = "--recompute" in sys.argv
if RECOMPUTE:
    sys.argv.remove("--recompute")

MODE = config.resolve_mode()
paths = config.get_mode_paths(MODE)
output_dir = paths["output_dir"]
summary_file = config.find_latest(output_dir, f"{paths['file_prefix']}_summary")
print(f"[{MODE}] Summary: {summary_file}")
df_summary = pd.read_csv(summary_file)

if MODE == "main":
    pipelines_lookup, _ = build_pipelines_main()
else:
    pipelines_lookup, _ = build_pipelines_shared_space(n_components=3)

n_perm = config.N_PERM_RESTRICTED

report_lines = ["=" * 78,
                f"GROUP/INDIVIDUAL LEVEL - RESTRICTED PERMUTATION TEST VS. CHANCE [{MODE.upper()}]",
                "=" * 78,
                f"N permutations: {n_perm} (pipeline refit per permutation, no grid search)",
                f"alpha = {config.ALPHA}  |  FDR method: {config.FDR_METHOD}",
                ""]

group_rows = []
individual_frames = {}

for cond in config.CONDITIONS:
    print(f"\n--- {cond} ---")
    fe_method_by_participant = modal_fe_method_per_participant(df_summary, cond)

    participant_data = {}
    for participant in fe_method_by_participant:
        X_path = os.path.join(paths["data_path"], participant, f"{participant}_{cond}_X.npy")
        y_path = os.path.join(paths["data_path"], participant, f"{participant}_{cond}_y.npy")
        if os.path.exists(X_path) and os.path.exists(y_path):
            participant_data[participant] = (np.load(X_path), np.load(y_path))

    block_size = config.BLOCK_SIZE_BY_CONDITION.get(cond) if MODE == "main" else None

    # One RNG per condition -> each condition is independently reproducible,
    # so loading from cache or recomputing a single condition gives the same results.
    rng = np.random.default_rng([config.RANDOM_STATE, config.CONDITIONS.index(cond)])

    # The permutations are the expensive part -> cache them; the cache is only
    # used if all settings and inputs match.
    cache_file = os.path.join(output_dir, f"{paths['file_prefix']}_{cond.lower()}_perm_cache.pkl")
    cache_key = {
        "n_perm": n_perm, "random_state": config.RANDOM_STATE, "block_size": block_size,
        "summary_file": os.path.basename(summary_file),
        "fe_method_by_participant": dict(fe_method_by_participant),
        "participants_with_data": sorted(participant_data),
    }
    cached = None
    if os.path.exists(cache_file) and not RECOMPUTE:
        with open(cache_file, "rb") as f:
            blob = pickle.load(f)
        if blob.get("key") == cache_key:
            cached = blob["result"]
            print(f"[{cond}] Permutations loaded from cache: {cache_file}")
        else:
            print(f"[{cond}] Cache does not match current settings/data -> recomputing.")

    if cached is None:
        cached = restricted_permutation_null_accuracies(
            participant_data, fe_method_by_participant, pipelines_lookup,
            n_perm=n_perm, rng=rng, outer_splits=5,
            random_state=config.RANDOM_STATE, block_size=block_size,
        )
        with open(cache_file, "wb") as f:
            pickle.dump({"key": cache_key, "result": cached}, f)
        print(f"[{cond}] Permutations saved: {cache_file}")

    participants, observed_accs, null_matrix, observed_classifier = cached

    # Null distribution as CSV: rows = permutations, columns = participants
    df_null = pd.DataFrame(null_matrix, columns=list(participants))
    df_null.index.name = "permutation"
    df_null.to_csv(os.path.join(output_dir, f"{paths['file_prefix']}_{cond.lower()}_null_distribution.csv"))

    # ---- Group level ----
    observed_group_mean, p_group = restricted_group_p_value(observed_accs, null_matrix)
    accs = np.array(list(observed_accs.values()))
    emp_chance_per_participant = null_matrix.mean(axis=0)
    d = cohens_d_paired(accs - emp_chance_per_participant)
    rrb = rank_biserial(accs - emp_chance_per_participant)
    n_above_theoretical = int((accs > config.CHANCE_LEVEL).sum())
    prop_above_theoretical = float((accs > config.CHANCE_LEVEL).mean())

    # Empirical chance level = mean of the permutation null distribution. It can
    # differ from the theoretical 0.5 (class imbalance, trial autocorrelation,
    # pipeline bias). The significance tests already use the empirical null
    # distribution; the empirical level is reported for the plain numerical comparison.
    empirical_chance_group = float(null_matrix.mean(axis=1).mean())
    n_above_empirical = int((accs > empirical_chance_group).sum())
    prop_above_empirical = float((accs > empirical_chance_group).mean())

    group_rows.append({
        "condition": cond, "n_participants": len(participants),
        "observed_group_mean_accuracy": observed_group_mean,
        "theoretical_chance_level": config.CHANCE_LEVEL,
        "empirical_chance_level": empirical_chance_group,
        "p_value_vs_chance": p_group,
        "cohens_d_vs_chance": d, "effect_size_words": effect_size_words(d),
        "rank_biserial_r_vs_chance": rrb,
        "n_participants_above_theoretical_chance": n_above_theoretical,
        "prop_participants_above_theoretical_chance": prop_above_theoretical,
        "n_participants_above_empirical_chance": n_above_empirical,
        "prop_participants_above_empirical_chance": prop_above_empirical,
        "significant_at_alpha": p_group < config.ALPHA,
    })

    # ---- Individual level ----
    p_individual = restricted_individual_p_values(participants, observed_accs, null_matrix)
    df_ind = pd.DataFrame([
        {"participant": p, "accuracy": observed_accs[p],
         "fe_method": fe_method_by_participant[p],
         "classifier": observed_classifier[p],
         "empirical_chance_level": null_matrix[:, j].mean(),
         "p_value_vs_chance": p_individual[p]}
        for j, p in enumerate(participants)
    ])
    reject, p_fdr = fdr_correct(df_ind["p_value_vs_chance"].values, alpha=config.ALPHA, method=config.FDR_METHOD)
    df_ind["p_value_vs_chance_fdr"] = p_fdr
    df_ind["significant_fdr"] = reject
    df_ind["above_theoretical_chance"] = df_ind["accuracy"] > config.CHANCE_LEVEL
    df_ind["above_empirical_chance"] = df_ind["accuracy"] > df_ind["empirical_chance_level"]
    individual_frames[cond] = df_ind
    df_ind.to_csv(os.path.join(output_dir, f"{paths['file_prefix']}_{cond.lower()}_individual_vs_chance.csv"), index=False)

    verdict = "SIGNIFICANTLY above chance" if p_group < config.ALPHA else "NOT significantly above chance"
    n_sig_ind = int(df_ind["significant_fdr"].sum())
    report_lines += [
        "-" * 78,
        f"{cond} (n={len(participants)}):",
        f"  Group mean accuracy: {observed_group_mean:.3f}",
        f"  Theoretical chance level: {config.CHANCE_LEVEL:.3f}  |  "
        f"Empirical chance level (permutation null mean): {empirical_chance_group:.3f}",
        f"  Group-level p-value vs. chance (restricted permutation, n={n_perm}): {p_group:.4f}",
        f"  Effect size: d = {d:.3f} ({effect_size_words(d)}); rank-biserial r = {rrb:.3f}",
        f"  {n_above_theoretical}/{len(participants)} ({100*prop_above_theoretical:.0f}%) "
        f"numerically above THEORETICAL chance ({config.CHANCE_LEVEL})",
        f"  {n_above_empirical}/{len(participants)} ({100*prop_above_empirical:.0f}%) "
        f"numerically above EMPIRICAL chance (permutation null mean)",
        f"  => {cond} {verdict}",
        f"  Individual level: {n_sig_ind}/{len(participants)} ({100*n_sig_ind/len(participants):.0f}%) "
        f"participants significantly above chance (FDR-corrected)",
        "",
    ]

# Group-level table
group_file = os.path.join(output_dir, f"{paths['file_prefix']}_group_vs_chance_stats.csv")
pd.DataFrame(group_rows).to_csv(group_file, index=False)

# ---- Individual level: overlap / asymmetry between SD and ID ----
if "SD" in individual_frames and "ID" in individual_frames:
    sd_df = individual_frames["SD"].set_index("participant")
    id_df = individual_frames["ID"].set_index("participant")
    common = sd_df.index.intersection(id_df.index)

    sd_sig = sd_df.loc[common, "significant_fdr"].astype(bool)
    id_sig = id_df.loc[common, "significant_fdr"].astype(bool)

    both = int((sd_sig & id_sig).sum())
    only_sd = int((sd_sig & ~id_sig).sum())
    only_id = int((~sd_sig & id_sig).sum())
    neither = int((~sd_sig & ~id_sig).sum())
    n_sd_sig, n_id_sig = int(sd_sig.sum()), int(id_sig.sum())

    pd.DataFrame(
        {"ID significant": [both, only_id], "ID not significant": [only_sd, neither]},
        index=["SD significant", "SD not significant"],
    ).to_csv(os.path.join(output_dir, f"{paths['file_prefix']}_sd_id_overlap.csv"))

    report_lines += ["-" * 78,
                     "Individual level, continued - overlap/asymmetry between SD and ID (individual level)",
                     "-" * 78]
    if n_sd_sig > 0:
        report_lines.append(f"Overlap: {both}/{n_sd_sig} ({100*both/n_sd_sig:.0f}%) of participants with "
                            f"above-chance accuracy in SD were also above chance in ID.")
    if n_id_sig > 0:
        report_lines.append(f"Asymmetry: of the {n_id_sig} participants with above-chance accuracy in ID, "
                            f"{both} ({100*both/n_id_sig:.0f}%) were above chance in SD.")
    report_lines.append("")

report_lines += ["=" * 78, "END OF REPORT", "=" * 78]
report_text = "\n".join(report_lines)
report_file = os.path.join(output_dir, f"{paths['file_prefix']}_group_vs_chance_summary.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\n[{MODE}] Report saved: {report_file}")
