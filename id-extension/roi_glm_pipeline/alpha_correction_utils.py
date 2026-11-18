"""
Statistical helpers for the ROI GLM analyses: combination of two runs into a
single p-value per participant/channel, and multiple-comparison corrections.

Run combination
---------------
Each participant has two runs (RunA, RunB). Before any correction can be
applied, the two runs are reduced to one p-value per participant and channel
with Fisher's method, ``combine_two_run_pvalue_fisher()``. The two p-values are
pooled into one combined p-value. Fisher's method has no notion of direction,
so the runs are only combined if their T-values have the same sign; otherwise
the channel is counted as not significant (combined p = 1.0). Optionally the
shared direction must also match the physiologically expected one (HbO
increase, HbR decrease; see below).

Multiple-comparison correction
------------------------------
- Per-channel significance ("% of participants significant per channel") is
  descriptive and uncorrected (Fisher-combined p < ALPHA).
- ``bonferroni()``: Bonferroni correction. Used for "participants without any
  significant channel", where the family is "all channels of one channel set
  for one fixed participant" (applied separately for each participant).
- ``effective_num_tests()`` / ``effective_family_size()``: Li & Ji (2005)
  effective number of independent tests (M_eff), a less conservative family
  size for Bonferroni. Neighbouring channels are spatially close and
  correlated, so treating them as fully independent over-corrects. M_eff is
  estimated once per channel family (across participants) from the eigenvalues
  of the correlation matrix of the channels' T-values; it lies between 1
  (perfectly correlated) and the number of tests (independent, i.e. plain
  Bonferroni). Because the eigenvalues of a correlation matrix sum to the
  number of tests, M_eff is always a whole number. For this estimation only,
  the two runs' T-values are averaged (``combine_two_run_tvalue()``); the
  averaged values never enter a significance decision.

Expected activation direction
-----------------------------
A task-evoked activation shows HbO increasing (T > 0) and HbR decreasing
(T < 0). The GLM p-values are two-sided, so without a sign check an effect in
the opposite direction (HbO decrease / HbR increase) would be counted as
significant, although it is not a typical activation and may reflect
artefacts (motion, systemic effects). With ``REQUIRE_EXPECTED_DIRECTION`` set
to True such effects count as not significant. This is different from
``same_direction()``, which only checks that RunA and RunB agree with each
other. The check only applies to the "> Baseline" contrasts: the difference
contrast (Hand clenching > Finger tapping) has no physiologically expected
sign.

References
----------
Li J, Ji L (2005). Adjusting multiple testing in multilocus analyses using the
eigenvalues of a correlation matrix. Heredity 95(3):221-227.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Expected activation direction
# -----------------------------------------------------------------------------
# Set to False for a purely two-sided, direction-agnostic analysis (then state
# in the methods that direction was not checked).
REQUIRE_EXPECTED_DIRECTION = True
EXPECTED_T_SIGN = {"Oxy": 1, "DeOxy": -1}
CONTRASTS_WITH_EXPECTED_DIRECTION = {"Contrast1", "Contrast2"}


def expected_sign_for(chromo_label, contrast_key):
    """+1 (HbO) or -1 (HbR) if a direction restriction applies, otherwise None
    (check switched off, or a contrast without a defined expected direction)."""
    if not REQUIRE_EXPECTED_DIRECTION:
        return None
    if contrast_key not in CONTRASTS_WITH_EXPECTED_DIRECTION:
        return None
    return EXPECTED_T_SIGN[chromo_label]


# -----------------------------------------------------------------------------
# Combining the two runs
# -----------------------------------------------------------------------------
def combine_two_run_tvalue(t_run_a, t_run_b):
    """Mean of the two runs' T-values (falls back to whichever run is
    available; None if both are missing). For M_eff estimation ONLY."""
    if t_run_a is None and t_run_b is None:
        return None
    if t_run_a is None:
        return t_run_b
    if t_run_b is None:
        return t_run_a
    return (t_run_a + t_run_b) / 2.0


def same_direction(t_run_a, t_run_b):
    """True if both T-values have the same sign, False if they differ or if
    either is exactly 0 (not clear evidence in any direction), None if either
    is missing."""
    if t_run_a is None or t_run_b is None:
        return None
    if t_run_a == 0 or t_run_b == 0:
        return False
    return (t_run_a > 0) == (t_run_b > 0)


def fisher_combined_pvalue(p_run_a, p_run_b):
    """
    Fisher's method for two independent p-values testing the same hypothesis.

    X = -2 * (ln pA + ln pB) ~ chi-square(df=4) under H0, so
    combined_p = P(chi2_4 >= X) = exp(-X/2) * (1 + X/2)
    (closed-form survival function for 4 degrees of freedom; no SciPy needed).

    The method ignores direction: the caller must make sure both p-values
    stem from evidence pointing the same way. Use
    ``combine_two_run_pvalue_fisher()`` unless you have checked that yourself.
    """
    pa = max(min(p_run_a, 1.0), 1e-300)  # guard against log(0)
    pb = max(min(p_run_b, 1.0), 1e-300)
    x = -2.0 * (np.log(pa) + np.log(pb))
    return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))


def combine_two_run_pvalue_fisher(p_run_a, p_run_b, t_run_a, t_run_b, expected_sign=None):
    """
    Direction-aware combination of RunA/RunB evidence via Fisher's method.

    If the runs disagree in the sign of their T-values they contradict each
    other and are NOT combined: combined_p is forced to 1.0. If
    ``expected_sign`` is given (+1 HbO, -1 HbR; see ``expected_sign_for()``),
    the shared direction must also match it, otherwise combined_p is forced to
    1.0 as well. In both cases the data are present and informative, so the
    participant/channel stays in the denominator and in the family size of
    downstream corrections; it just cannot become significant.
    ``expected_sign=None`` disables the expected-direction check.

    Returns ``(combined_p, status)``:
        "combined"              same direction; combined_p is the Fisher p-value
        "conflicting_direction" runs disagree; combined_p = 1.0
        "wrong_direction"       runs agree but oppose expected_sign; combined_p = 1.0
        "missing"               p or T missing in >= 1 run; combined_p is None
    """
    if p_run_a is None or p_run_b is None or t_run_a is None or t_run_b is None:
        return None, "missing"
    if not same_direction(t_run_a, t_run_b):
        return 1.0, "conflicting_direction"
    if expected_sign is not None and (t_run_a > 0) != (expected_sign > 0):
        return 1.0, "wrong_direction"
    return fisher_combined_pvalue(p_run_a, p_run_b), "combined"


# -----------------------------------------------------------------------------
# Multiple-comparison corrections
# -----------------------------------------------------------------------------
def bonferroni(pvalue_dict, family_size=None):
    """
    Bonferroni correction.

    pvalue_dict: {key: p_value or None}
    family_size: number of tests in the family; may be a float (e.g. M_eff).
                 Defaults to the number of non-missing p-values.
    Returns:     {key: adjusted_p_value or None}, capped at 1.0
    """
    non_missing = [p for p in pvalue_dict.values() if p is not None]
    n = family_size if family_size is not None else len(non_missing)
    return {
        k: (None if p is None or n == 0 else min(p * n, 1.0))
        for k, p in pvalue_dict.items()
    }


def effective_num_tests(corr_matrix):
    """
    Li & Ji (2005) effective number of independent tests, M_eff, from the
    correlation matrix of M correlated tests.

    M_eff = sum_i f(lambda_i) over the eigenvalues lambda_i of the
    correlation matrix, with
    f(lambda) = I(lambda >= 1) + (lambda - floor(lambda)),
    where I() is the indicator function (1 if true, else 0).

    M_eff lies in [1, M]: M for independent tests (identity matrix), 1 for
    perfectly correlated tests.
    """
    eigvals = np.linalg.eigvalsh(np.asarray(corr_matrix, dtype=float))
    eigvals = np.clip(eigvals, 0.0, None)  # remove tiny negative values (floating-point noise)
    contributions = np.where(eigvals >= 1.0, 1.0 + (eigvals - np.floor(eigvals)), eigvals)
    return float(np.sum(contributions))


def effective_family_size(tvalue_dict, channels, min_complete_cases=None):
    """
    Estimate M_eff for a family of ``channels`` from the across-participant
    correlation matrix of their T-values.

    tvalue_dict: {pid: {channel: t_value or None}}, typically run-averaged via
                 ``combine_two_run_tvalue()``.
    channels:    ordered channel keys defining the family (at least 2).
    min_complete_cases: minimum number of participants with a T-value on EVERY
                 channel (list-wise deletion) needed to trust the correlation
                 estimate; default max(10, len(channels) + 2). Below that,
                 M_eff falls back to len(channels), i.e. plain Bonferroni.

    Returns ``(m_eff, n_participants_used)``.
    """
    m = len(channels)
    if min_complete_cases is None:
        min_complete_cases = max(10, m + 2)

    rows = []
    for chvals in tvalue_dict.values():
        vals = [chvals.get(ch) for ch in channels]
        if all(v is not None for v in vals):
            rows.append(vals)
    n_used = len(rows)

    if n_used < min_complete_cases:
        return float(m), n_used

    corr = np.corrcoef(np.array(rows, dtype=float), rowvar=False)
    m_eff = max(1.0, min(effective_num_tests(corr), float(m)))
    return m_eff, n_used
