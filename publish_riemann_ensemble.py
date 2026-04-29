#!/usr/bin/env python
"""
publish_riemann_ensemble.py
============================
Cross-subject fNIRS task classification with a four-base Riemannian
ensemble on SPD matrices from the post-baseline epoch.

Pipeline
--------
- 109 participants, 26 channels (13 HbO + 13 HbR), 0..+20 s post-stim
  window, ~10.2 Hz sampling.
- Four diverse Riemannian bases (SCM+shrinkage, Ledoit-Wolf SCM, RBF
  kernel matrix, slow-band median-bin cospectrum) feed Tangent-Space
  classifiers (LogReg, LDA-shrink), with time-delay embedding at
  0 / 5.9 / 7.8 s.
- 5-fold GroupKFold by participant (deterministic, no shuffle).
- Ensemble: simple (unweighted) mean of per-trial OOF probabilities.
- Optional within-participant label-permutation test gated by
  ``N_PERMUTATIONS``.

Outputs (``results/publish_ensemble/``): per-trial scores, per-fold
accuracies, summary table, optional permutation results, and PNG plots.

Implementation: pyriemann (Barachant et al., JMLR 2014) and scikit-learn.
Set ``KMP_DUPLICATE_LIB_OK=TRUE`` on macOS (Conda OpenMP workaround).
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline

from pyriemann.estimation import Covariances, Shrinkage, Kernels, CoSpectra
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "data"
PUBLISH_OUTPUT_DIR = "results/publish_ensemble"
CONDITION = "SD"

SAMPLING_RATE_HZ = 10.2
TIME_WINDOW = (-5.0, 25.0)
CROP_T0, CROP_T1 = 0.0, 20.0

HBO_CHANNELS = list(range(0, 13))
HBR_CHANNELS = list(range(13, 26))
CONCAT = HBO_CHANNELS + HBR_CHANNELS

N_CV_FOLDS = 5
N_PERMUTATIONS = 500
PERMUTATION_SEED = 42

SPECTRAL_WINDOW = 64
SPECTRAL_OVERLAP = 0.75
FREQUENCY_BANDS_HZ = {
    "slow": (0.01, 0.5),
    "mid":  (0.5,  1.0),
    "high": (1.0,  1.6),
}

# Four Riemannian base models, search-selected for ensemble diversity.
BASE_CONFIGS = [
    {"name": "scm_shr20_d80_ts_lr",
     "channels": CONCAT, "delay_samples": 80,
     "estimator": {"kind": "cov",
                    "params": {"estimator": "scm", "shrinkage": 0.20}},
     "classifier": "ts_lr"},
    {"name": "lwf_d60_ts_lr",
     "channels": CONCAT, "delay_samples": 60,
     "estimator": {"kind": "cov", "params": {"estimator": "lwf"}},
     "classifier": "ts_lr"},
    {"name": "rbf_d60_ts_lr",
     "channels": CONCAT, "delay_samples": 60,
     "estimator": {"kind": "kern", "params": {"metric": "rbf"}},
     "classifier": "ts_lr"},
    {"name": "cospslow_ts_lda",
     "channels": CONCAT, "delay_samples": 0,
     "estimator": {"kind": "cosp", "params": {"bands": ["slow"]}},
     "classifier": "ts_lda"},
]

NAMED_ENSEMBLES = {
    "simple_mean": {"members": [c["name"] for c in BASE_CONFIGS]},
}


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(data_path, condition):
    """Stack per-participant npy files into ``(X, y, groups)``."""
    participants = sorted(
        p for p in os.listdir(data_path)
        if p.startswith("P") and os.path.isdir(os.path.join(data_path, p))
    )
    X_list, y_list, g_list = [], [], []
    for p in participants:
        Xp = os.path.join(data_path, p, f"{p}_{condition}_X.npy")
        yp = os.path.join(data_path, p, f"{p}_{condition}_y.npy")
        if not (os.path.exists(Xp) and os.path.exists(yp)):
            continue
        X_list.append(np.load(Xp))
        y_list.append(np.load(yp).astype(int))
        g_list.extend([p] * len(y_list[-1]))
    X = np.concatenate(X_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list)
    classes = np.unique(y)
    return X, (y == classes[1]).astype(int), np.array(g_list)


def crop_window(X, t_start_full, t_end_full, t_keep_start, t_keep_end):
    """Crop the time axis to ``[t_keep_start, t_keep_end)``."""
    t = np.linspace(t_start_full, t_end_full, X.shape[-1], endpoint=False)
    keep = (t >= t_keep_start) & (t < t_keep_end)
    return X[..., keep], t[keep]


# ============================================================================
# SPD construction
# ============================================================================

def time_delay_augment(X, delay):
    """Channel-axis stack of X with a copy delayed by ``delay`` samples."""
    if delay <= 0:
        return X
    return np.concatenate([X[:, :, delay:], X[:, :, :-delay]], axis=1)


def _stack_block_diagonal(per_band):
    n, ch, _ = per_band[0].shape
    nb = len(per_band)
    out = np.zeros((n, ch * nb, ch * nb), dtype=per_band[0].dtype)
    for i, M in enumerate(per_band):
        out[:, i * ch:(i + 1) * ch, i * ch:(i + 1) * ch] = M
    return out


def estimate_spd(X, kind, params, ridge=1e-6):
    """Build (n, d, d) SPD matrices via cov / kern / cosp."""
    if kind == "cov":
        params = dict(params)
        alpha = params.pop("shrinkage", None)
        C = Covariances(**params).fit_transform(X)
        if alpha is not None:
            C = Shrinkage(shrinkage=alpha).fit_transform(C)
    elif kind == "kern":
        C = Kernels(**params).fit_transform(X)
    elif kind == "cosp":
        per_band = []
        for band in params["bands"]:
            fmin, fmax = FREQUENCY_BANDS_HZ[band]
            C4 = CoSpectra(window=SPECTRAL_WINDOW, overlap=SPECTRAL_OVERLAP,
                            fmin=fmin, fmax=fmax,
                            fs=SAMPLING_RATE_HZ).fit_transform(X)
            per_band.append(C4[..., C4.shape[-1] // 2])
        C = (per_band[0] if len(per_band) == 1
             else _stack_block_diagonal(per_band))
    else:
        raise ValueError(f"Unknown estimator kind: {kind!r}")
    C = (C + C.transpose(0, 2, 1)) / 2
    C += ridge * np.eye(C.shape[-1])[np.newaxis, :, :]
    return C


def build_spd_for_config(X, cfg):
    Xs = X[:, cfg["channels"], :]
    Xs = time_delay_augment(Xs, cfg["delay_samples"])
    return estimate_spd(Xs, cfg["estimator"]["kind"],
                          cfg["estimator"]["params"])


# ============================================================================
# Cross-validation and base classifiers
# ============================================================================

def grouped_cv_splits(y, groups, n_splits):
    """Deterministic GroupKFold splits (no shuffle, no random_state)."""
    return list(GroupKFold(n_splits=n_splits)
                .split(np.zeros(len(y)), y, groups))


def precompute_fold_features(spd_blocks, splits):
    """Per-fold tangent-space projection per base config (y-independent;
    reused across the main run and every permutation)."""
    cache = {}
    for name, C in spd_blocks.items():
        per_fold = []
        for tr, te in splits:
            ts = TangentSpace(metric="logeuclid")
            per_fold.append({
                "V_tr": ts.fit_transform(C[tr]),
                "V_te": ts.transform(C[te]),
            })
        cache[name] = per_fold
    return cache


def _make_classifier(name):
    """Tangent-space classifier factory."""
    if name == "ts_lr":
        return LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    if name == "ts_lda":
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto")),
        ])
    raise ValueError(f"Unknown classifier: {name!r}")


def cv_scores(per_fold, y, classifier, splits):
    """Per-trial OOF probability scores from precomputed fold features."""
    scores = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(splits):
        f = per_fold[fi]
        clf = _make_classifier(classifier)
        clf.fit(f["V_tr"], y[tr])
        scores[te] = clf.predict_proba(f["V_te"])[:, 1]
    return scores


# ============================================================================
# Ensemble + permutation test
# ============================================================================

def _ensemble_for(all_scores, ens_cfg, name_to_idx):
    """Simple (unweighted) mean of base OOF probabilities."""
    idxs = [name_to_idx[m] for m in ens_cfg["members"]]
    return all_scores[idxs].mean(axis=0)


def per_fold_acc(scores, y, splits):
    return np.array([
        balanced_accuracy_score(y[te], (scores[te] > 0.5).astype(int))
        for _, te in splits
    ])


def permute_within_participants(y, groups, rng):
    """Shuffle labels within each participant; preserves GroupKFold structure."""
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def run_permutation_test(cache, y, groups, splits, n_perm, seed):
    """Within-participant label-shuffle null for each named ensemble."""
    rng = np.random.default_rng(seed)
    name_to_idx = {c["name"]: i for i, c in enumerate(BASE_CONFIGS)}
    null = {n: np.zeros(n_perm) for n in NAMED_ENSEMBLES}
    for p in tqdm(range(n_perm), desc="Permutations"):
        y_perm = permute_within_participants(y, groups, rng)
        all_s = np.zeros((len(BASE_CONFIGS), len(y)))
        for i, cfg in enumerate(BASE_CONFIGS):
            all_s[i] = cv_scores(
                cache[cfg["name"]], y_perm, cfg["classifier"], splits)
        for ens_name, ens_cfg in NAMED_ENSEMBLES.items():
            avg = _ensemble_for(all_s, ens_cfg, name_to_idx)
            null[ens_name][p] = balanced_accuracy_score(
                y_perm, (avg > 0.5).astype(int))
    return null


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(PUBLISH_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # -- Load and crop --
    X_full, y, groups = load_dataset(DATA_PATH, CONDITION)
    X, _ = crop_window(X_full, *TIME_WINDOW, CROP_T0, CROP_T1)
    splits = grouped_cv_splits(y, groups, N_CV_FOLDS)
    print(f"Loaded {len(y)} trials, {len(set(groups))} participants, "
          f"X={X.shape}", flush=True)
    print(f"GroupKFold ({N_CV_FOLDS}-fold): "
          f"test sizes = {[len(te) for _, te in splits]}", flush=True)

    # -- SPDs and per-fold tangent vectors --
    spd = {cfg["name"]: build_spd_for_config(X, cfg)
            for cfg in tqdm(BASE_CONFIGS, desc="Build SPDs")}
    cache = precompute_fold_features(spd, splits)
    print("SPD dimensions:")
    for cfg in BASE_CONFIGS:
        print(f"  {cfg['name']:24s} {spd[cfg['name']].shape}")

    # -- Base CV --
    all_scores = np.zeros((len(BASE_CONFIGS), len(y)))
    base_accs, base_fold_accs = {}, {}
    for i, cfg in enumerate(tqdm(BASE_CONFIGS, desc="Fit base models")):
        s = cv_scores(cache[cfg["name"]], y, cfg["classifier"], splits)
        all_scores[i] = s
        base_accs[cfg["name"]] = balanced_accuracy_score(
            y, (s > 0.5).astype(int))
        base_fold_accs[cfg["name"]] = per_fold_acc(s, y, splits)

    # -- Ensembles --
    name_to_idx = {c["name"]: i for i, c in enumerate(BASE_CONFIGS)}
    ens_accs, ens_fold_accs = {}, {}
    print("\nNamed ensembles", flush=True)
    for ens_name, ens_cfg in NAMED_ENSEMBLES.items():
        avg = _ensemble_for(all_scores, ens_cfg, name_to_idx)
        ens_accs[ens_name] = balanced_accuracy_score(
            y, (avg > 0.5).astype(int))
        ens_fold_accs[ens_name] = per_fold_acc(avg, y, splits)
        print(f"  {ens_name:20s} {ens_accs[ens_name]:.4f}", flush=True)

    # -- Tables --
    scores_csv = os.path.join(PUBLISH_OUTPUT_DIR,
                                f"per_trial_scores_{ts}.csv")
    (pd.DataFrame(all_scores.T,
                    columns=[c["name"] for c in BASE_CONFIGS])
     .assign(y_true=y, participant=groups)
     .to_csv(scores_csv, index=False))

    per_fold_rows = [{"name": n, "fold": k, "bal_acc": ens_fold_accs[n][k]}
                      for n in NAMED_ENSEMBLES for k in range(N_CV_FOLDS)]
    per_fold_csv = os.path.join(PUBLISH_OUTPUT_DIR,
                                 f"per_fold_results_{ts}.csv")
    pd.DataFrame(per_fold_rows).to_csv(per_fold_csv, index=False)

    summary_rows = []
    for cfg in BASE_CONFIGS:
        fa = base_fold_accs[cfg["name"]]
        summary_rows.append({
            "name": cfg["name"], "kind": "base",
            "bal_acc_overall": base_accs[cfg["name"]],
            "fold_mean": fa.mean(), "fold_std": fa.std(),
            "fold_min": fa.min(), "fold_max": fa.max(),
        })
    for n in NAMED_ENSEMBLES:
        fa = ens_fold_accs[n]
        summary_rows.append({
            "name": n, "kind": "ensemble",
            "bal_acc_overall": ens_accs[n],
            "fold_mean": fa.mean(), "fold_std": fa.std(),
            "fold_min": fa.min(), "fold_max": fa.max(),
        })
    summary_df = (pd.DataFrame(summary_rows)
                    .sort_values(["kind", "bal_acc_overall"],
                                  ascending=[True, False]))
    summary_csv = os.path.join(PUBLISH_OUTPUT_DIR,
                                 f"summary_results_{ts}.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -- Console summary --
    print("\n" + "=" * 78)
    print(f"SUMMARY ({N_CV_FOLDS}-fold GroupKFold, "
          f"n={len(set(groups))} participants)")
    print("=" * 78)
    print(f"  {'name':<22s}  {'kind':<8s}  {'overall':>7s}  "
          f"{'mu':>6s}+/-{'sigma':<6s}  {'min':>6s}  {'max':>6s}")
    for _, r in summary_df.iterrows():
        print(f"  {r['name']:<22s}  {r['kind']:<8s}  "
              f"{r['bal_acc_overall']:>7.4f}  "
              f"{r['fold_mean']:>6.4f}+/-{r['fold_std']:<6.4f}  "
              f"{r['fold_min']:>6.4f}  {r['fold_max']:>6.4f}")

    # -- Optional permutation test --
    perm_csv = null_csv = perm_png = None
    null_accs = {n: np.array([]) for n in NAMED_ENSEMBLES}
    perm_df = pd.DataFrame()
    if N_PERMUTATIONS > 0:
        print(f"\nPermutation test ({N_PERMUTATIONS} within-participant "
              f"shuffles)...", flush=True)
        null_accs = run_permutation_test(
            cache, y, groups, splits, N_PERMUTATIONS, PERMUTATION_SEED)
        perm_rows = []
        for ens_name, observed in ens_accs.items():
            null = null_accs[ens_name]
            # Add-one Laplace: minimum p = 1/(N+1)
            p = (1 + np.sum(null >= observed)) / (len(null) + 1)
            perm_rows.append({
                "name": ens_name, "observed": observed,
                "null_mean": float(null.mean()),
                "null_std":  float(null.std()),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "p_value": float(p), "n_perm": int(len(null)),
            })
        perm_df = (pd.DataFrame(perm_rows)
                   .sort_values("observed", ascending=False))
        perm_csv = os.path.join(PUBLISH_OUTPUT_DIR,
                                  f"permutation_results_{ts}.csv")
        perm_df.to_csv(perm_csv, index=False)
        null_long = pd.DataFrame({
            "name": np.repeat(list(null_accs.keys()), N_PERMUTATIONS),
            "perm": np.tile(np.arange(N_PERMUTATIONS), len(null_accs)),
            "bal_acc": np.concatenate([null_accs[n] for n in null_accs]),
        })
        null_csv = os.path.join(
            PUBLISH_OUTPUT_DIR,
            f"permutation_null_distribution_{ts}.csv")
        null_long.to_csv(null_csv, index=False)
        print(f"  {'name':<20s}  {'obs':>6s}  "
              f"{'mu':>6s}+/-{'sigma':<6s}  {'p':>6s}")
        for _, r in perm_df.iterrows():
            print(f"  {r['name']:<20s}  {r['observed']:>6.4f}  "
                  f"{r['null_mean']:>6.4f}+/-{r['null_std']:<6.4f}  "
                  f"{r['p_value']:>6.4g}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # Bar plot: bases (blue) + ensembles (red)
    plot_df = pd.concat([
        summary_df[summary_df["kind"] == "base"]
            .sort_values("bal_acc_overall"),
        summary_df[summary_df["kind"] == "ensemble"]
            .sort_values("bal_acc_overall"),
    ]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    accs = plot_df["bal_acc_overall"].tolist()
    colors = ["#e74c3c" if k == "ensemble" else "#3498db"
                for k in plot_df["kind"]]
    x = np.arange(len(plot_df))
    ax.bar(x, accs, color=colors, alpha=0.85,
            yerr=plot_df["fold_std"], capsize=4)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["name"], rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy (overall +/- per-fold SD)")
    ax.set_title(f"Riemannian-ensemble cross-subject classification - "
                  f"{N_CV_FOLDS}-fold GroupKFold (n={len(set(groups))})")
    ax.set_ylim(0.5, max(0.62, max(accs) + 0.02))
    for i, v in enumerate(accs):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.legend(handles=[
        Patch(facecolor="#3498db", label="base model"),
        Patch(facecolor="#e74c3c", label="ensemble"),
        plt.Line2D([], [], color="gray", ls="--", label="chance"),
    ], loc="lower left", fontsize=9)
    fig.tight_layout()
    summary_png = os.path.join(PUBLISH_OUTPUT_DIR,
                                f"ensemble_summary_{ts}.png")
    fig.savefig(summary_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ROC curves: each base (faded) + each ensemble (bold), AUC in legend
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, cfg in enumerate(BASE_CONFIGS):
        fpr, tpr, _ = roc_curve(y, all_scores[i])
        ax.plot(fpr, tpr, color="#3498db", alpha=0.45, linewidth=1.2,
                 label=f"{cfg['name']} (AUC={auc(fpr, tpr):.3f})")
    ens_palette = ["#2980b9", "#e74c3c"]
    for j, (n, ens_cfg) in enumerate(NAMED_ENSEMBLES.items()):
        avg = _ensemble_for(all_scores, ens_cfg, name_to_idx)
        fpr, tpr, _ = roc_curve(y, avg)
        ax.plot(fpr, tpr, color=ens_palette[j % len(ens_palette)],
                 linewidth=2.4, alpha=0.95,
                 label=f"{n} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves ({N_CV_FOLDS}-fold OOF predictions)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    roc_png = os.path.join(PUBLISH_OUTPUT_DIR, f"roc_curves_{ts}.png")
    fig.savefig(roc_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Per-base OOF score correlation heatmap (visualises ensemble diversity:
    # lower off-diagonal correlations -> more orthogonal base information).
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    corr = np.corrcoef(all_scores)
    names = [c["name"] for c in BASE_CONFIGS]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color=("white" if abs(corr[i, j]) > 0.6 else "black"),
                     fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Per-base OOF score correlation")
    fig.tight_layout()
    corr_png = os.path.join(PUBLISH_OUTPUT_DIR,
                              f"base_score_correlation_{ts}.png")
    fig.savefig(corr_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Permutation null distributions
    if N_PERMUTATIONS > 0:
        n_ens = len(NAMED_ENSEMBLES)
        fig, axes = plt.subplots(1, n_ens, figsize=(4 * n_ens, 4),
                                  sharey=True, squeeze=False)
        palette = ["#2980b9", "#e74c3c"]
        for ax, (j, name) in zip(axes.flat,
                                   enumerate(perm_df["name"].tolist())):
            null = null_accs[name]
            obs = ens_accs[name]
            p = perm_df[perm_df["name"] == name]["p_value"].iloc[0]
            ax.hist(null, bins=30, color="#bdc3c7", edgecolor="white",
                     alpha=0.85)
            ax.axvline(obs, color=palette[j % len(palette)], linewidth=2.5,
                        label=f"observed = {obs:.4f}")
            ax.axvline(0.5, color="gray", ls="--", alpha=0.4, label="chance")
            ax.set_xlabel("Balanced accuracy")
            if j == 0:
                ax.set_ylabel(f"Frequency (n_perm = {N_PERMUTATIONS})")
            ax.set_title(f"{name}\np = {p:.4g}", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Within-participant label-permutation nulls")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        perm_png = os.path.join(PUBLISH_OUTPUT_DIR,
                                 f"permutation_test_{ts}.png")
        fig.savefig(perm_png, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("\nSaved:")
    for path in (scores_csv, per_fold_csv, summary_csv,
                  perm_csv, null_csv,
                  summary_png, roc_png, corr_png, perm_png):
        if path:
            print(f"  {path}")


if __name__ == "__main__":
    main()
