"""
core/stats_core.py
------------------
Statistical building blocks shared across the pipeline:
permutation tests (vs. chance, paired, group-level), effect sizes
(Cohen's d, rank-biserial r), multiple-comparison correction, and
effect-size labels. Kept in one place so that the methodology cannot
silently diverge between scripts.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------
# Effect sizes
# ------------------------------------------------------------------

def rank_biserial(diffs):
    """
    Matched-pairs / one-sample rank-biserial correlation against a reference
    of 0 (non-parametric counterpart of Cohen's d). Range: -1 to 1.
    Differences of exactly 0 are dropped.
    """
    diffs = np.asarray(diffs)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return np.nan
    ranks = rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    return (r_plus - r_minus) / ranks.sum()


def cohens_d_one_sample(values, reference):
    """Cohen's d of `values` against a fixed reference value."""
    values = np.asarray(values)
    sd = values.std(ddof=1)
    if sd == 0:
        return np.nan
    return (values.mean() - reference) / sd


def cohens_d_paired(diffs):
    """Cohen's d_z for paired data, computed from the per-pair differences."""
    diffs = np.asarray(diffs)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return np.nan
    return diffs.mean() / sd


def effect_size_words(d):
    """Conventional Cohen (1988) labels applied to |d|."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    return "large"


# ------------------------------------------------------------------
# Permutation tests
# ------------------------------------------------------------------

def laplace_p_value(n_null_at_least_as_extreme, n_perm):
    """
    Add-one corrected permutation p-value: (1 + #{null >= observed}) / (n_perm + 1).
    Avoids p = 0 (the smallest justifiable statement is p < 1/(n_perm+1));
    standard practice for permutation tests (e.g., North et al., 2002).
    """
    return (1 + n_null_at_least_as_extreme) / (n_perm + 1)


def permutation_test_vs_chance(y_true, y_pred, n_perm, rng):
    """
    One-sided post-hoc permutation test: is the accuracy above chance?
    Trial labels are shuffled and compared with the fixed predictions.
    Returns (accuracy, p_value).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = (y_true == y_pred).mean()
    perm_accs = np.empty(n_perm)
    for i in range(n_perm):
        perm_accs[i] = (rng.permutation(y_true) == y_pred).mean()
    return acc, laplace_p_value(int((perm_accs >= acc).sum()), n_perm)


def paired_permutation_test(corr_a, corr_b, n_perm, rng):
    """
    One-sided paired permutation test (random label swap within each pair):
    is mean(b - a) > 0? corr_a / corr_b are 0/1 correctness arrays; if their
    lengths differ they are truncated to the shorter one.
    Returns (observed_diff, p_value).
    """
    corr_a = np.asarray(corr_a)
    corr_b = np.asarray(corr_b)
    n = min(len(corr_a), len(corr_b))
    corr_a, corr_b = corr_a[:n], corr_b[:n]

    obs_diff = (corr_b - corr_a).mean()
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        a_perm = np.where(swap, corr_b, corr_a)
        b_perm = np.where(swap, corr_a, corr_b)
        perm_diffs[i] = (b_perm - a_perm).mean()
    return obs_diff, laplace_p_value(int((perm_diffs >= obs_diff).sum()), n_perm)


def group_level_permutation_test(values_a, values_b, n_perm, rng):
    """
    Group-level paired permutation test on per-participant values (e.g.
    accuracy in condition A vs. B): random condition swap within each
    participant. Returns (observed_mean_diff [b - a], p_value).
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    obs_diff = (values_b - values_a).mean()

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        a_perm = np.where(swap, values_b, values_a)
        b_perm = np.where(swap, values_a, values_b)
        perm_diffs[i] = (b_perm - a_perm).mean()
    return obs_diff, laplace_p_value(int((perm_diffs >= obs_diff).sum()), n_perm)


def group_vs_chance_permutation_test(participant_arrays, chance_level, n_perm, rng):
    """
    Post-hoc group-level vs.-chance test: labels are shuffled within each
    participant, per-participant accuracy is recomputed against the fixed
    predictions, and the group mean is formed; repeated n_perm times.

    participant_arrays : {participant_id: (y_true, y_pred)}
    Returns (per_participant_acc [pd.Series], observed_group_mean, p_value).
    `chance_level` is accepted for API compatibility but not used; the null
    distribution comes from the permutations.
    """
    per_participant_acc = pd.Series({
        p: (np.asarray(yt) == np.asarray(yp)).mean()
        for p, (yt, yp) in participant_arrays.items()
    })
    observed_group_mean = per_participant_acc.mean()

    null_group_means = np.empty(n_perm)
    for i in range(n_perm):
        perm_accs = np.empty(len(participant_arrays))
        for idx, (p, (y_true, y_pred)) in enumerate(participant_arrays.items()):
            perm_accs[idx] = (rng.permutation(y_true) == y_pred).mean()
        null_group_means[i] = perm_accs.mean()

    p_value = laplace_p_value(int((null_group_means >= observed_group_mean).sum()), n_perm)
    return per_participant_acc, observed_group_mean, p_value


# ------------------------------------------------------------------
# p-values for the restricted permutation test
# ------------------------------------------------------------------
# These take the already computed observed/null accuracies from
# core.cv_core.restricted_permutation_null_accuracies, so no model is refit.

def restricted_group_p_value(observed_accs, null_matrix):
    """
    Group-level p-value: observed group-mean accuracy vs. the distribution of
    group means under label permutation. Returns (observed_group_mean, p_value).
    """
    n_perm = null_matrix.shape[0]
    observed_group_mean = float(np.mean(list(observed_accs.values())))
    null_group_means = null_matrix.mean(axis=1)
    n_exceed = int((null_group_means >= observed_group_mean).sum())
    return observed_group_mean, laplace_p_value(n_exceed, n_perm)


def restricted_individual_p_values(participants, observed_accs, null_matrix):
    """Individual-level p-values, one per participant. Returns {participant: p_value}."""
    n_perm = null_matrix.shape[0]
    return {
        p: laplace_p_value(int((null_matrix[:, j] >= observed_accs[p]).sum()), n_perm)
        for j, p in enumerate(participants)
    }


# ------------------------------------------------------------------
# Multiple comparisons
# ------------------------------------------------------------------

def fdr_correct(p_values, alpha=0.05, method="fdr_bh"):
    """Multiple-comparison correction (default Benjamini-Hochberg). Returns (reject, p_corrected)."""
    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return reject, p_corr
