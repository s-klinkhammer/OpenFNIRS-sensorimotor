"""
core/cv_core.py
---------------
Shared nested cross-validation logic used by 02_nested_cv.py for both the
main analysis (full pipeline search) and the sensitivity analysis (shared
xDAWN latent space), plus the building blocks of the restricted permutation
test used by 04_group_vs_chance.py.

Only the pipeline/grid definitions differ between the two analysis modes;
the CV loop itself is identical and therefore implemented once here.
"""

import os
import json
import copy
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances, XdawnCovariances, Shrinkage, BlockCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import SVC, MDM


# ------------------------------------------------------------------
# Pipeline builders
# ------------------------------------------------------------------

def _tangent_logreg():
    return Pipeline([("tangent", TangentSpace()),
                     ("logreg", LogisticRegression(max_iter=1000))])


def _chain(prefix_steps, clf_name, clf):
    """Append a classifier (single estimator or sub-pipeline) to a list of steps."""
    steps = list(prefix_steps)
    if isinstance(clf, Pipeline):
        steps += [(f"{clf_name}_{n}", t) for n, t in clf.steps]
    else:
        steps.append((clf_name, clf))
    return Pipeline(steps)


def build_pipelines_main():
    """
    Full pipeline search: 3 feature-extraction methods x 3 classifiers.

    Feature extraction: xDAWN covariances, plain covariances, and block
    covariances (HbO | HbR blocks). Classifiers: Riemannian SVC,
    tangent-space logistic regression, and MDM.

    Returns (pipelines, param_grids), both dicts keyed by pipeline name
    ("<feature>_<classifier>").
    """
    estimators = {
        "xdawn": [("xdawn", XdawnCovariances(nfilter=6, estimator="lwf")),
                  ("shrinkage", Shrinkage())],
        "covariances": [("cov", Covariances(estimator="lwf")),
                        ("shrinkage", Shrinkage())],
        # block_size is a placeholder; it is overwritten per condition
        # (see run_nested_cv / _prepare_fixed_pipeline).
        "block_covariances": [("cov", BlockCovariances(block_size=13, estimator="lwf")),
                              ("shrinkage", Shrinkage())],
    }
    classifiers = {
        "svc": SVC(metric="riemann"),
        "tangent_logreg": _tangent_logreg(),
        "mdm": MDM(metric="riemann"),
    }

    pipelines, param_grids = {}, {}
    for est_name, est_steps in estimators.items():
        prefixed = [(f"{est_name}_{n}", t) for n, t in est_steps]
        for clf_name, clf in classifiers.items():
            pipe_name = f"{est_name}_{clf_name}"
            pipelines[pipe_name] = _chain(prefixed, clf_name, clf)

            grid = {}
            if est_name == "xdawn":
                grid[f"{est_name}_xdawn__nfilter"] = [6, 8, 10]
                grid[f"{est_name}_xdawn__estimator"] = ["corr", "lwf", "oas"]
            else:
                grid[f"{est_name}_cov__estimator"] = ["corr", "lwf", "oas"]
            grid[f"{est_name}_shrinkage__shrinkage"] = [0.1, 0.2, 0.5]

            if clf_name == "svc":
                grid["svc__C"] = [0.01, 0.1, 1]
            elif clf_name == "tangent_logreg":
                grid["tangent_logreg_logreg__C"] = [0.01, 0.1, 1]
            elif clf_name == "mdm":
                grid["mdm__metric"] = ["logeuclid", "riemann", "wasserstein"]
            param_grids[pipe_name] = grid

    return pipelines, param_grids


def build_pipelines_shared_space(n_components=3):
    """
    Sensitivity analysis: xDAWN projection with a fixed number of components
    per class (identical latent dimensionality for SD and ID), combined with
    the same three classifiers as in the main analysis.

    Returns (pipelines, param_grids) keyed by "xdawn_<classifier>".
    """
    classifiers = {
        "tangent_logreg": _tangent_logreg(),
        "mdm": MDM(metric="riemann"),
        "svc": SVC(metric="riemann"),
    }
    pipelines, param_grids = {}, {}
    for clf_name, clf in classifiers.items():
        pipe_name = f"xdawn_{clf_name}"
        prefix = [("xdawn", XdawnCovariances(nfilter=n_components, estimator="lwf")),
                  ("shrinkage", Shrinkage())]
        pipelines[pipe_name] = _chain(prefix, clf_name, clf)

        grid = {"xdawn__estimator": ["corr", "lwf", "oas"],
                "shrinkage__shrinkage": [0.1, 0.2, 0.5]}
        if clf_name == "svc":
            grid["svc__C"] = [0.01, 0.1, 1]
        elif clf_name == "tangent_logreg":
            grid["tangent_logreg_logreg__C"] = [0.01, 0.1, 1]
        elif clf_name == "mdm":
            grid["mdm__metric"] = ["logeuclid", "riemann", "wasserstein"]
        param_grids[pipe_name] = grid

    return pipelines, param_grids


def check_block_size(X, block_size, context=""):
    """
    Guard against silent errors in block_covariances: the number of channels
    in X must be a multiple of block_size (expected: 2 blocks = HbO | HbR).
    The channel order itself is checked in 01_preprocess_data.py.
    """
    if block_size is None:
        return
    n_ch = X.shape[1]
    if n_ch % block_size != 0:
        raise ValueError(
            f"{context}block_size={block_size} does not match the number of channels in X ({n_ch}). "
            f"Check config.BLOCK_SIZE_BY_CONDITION and the channel selection in 01_preprocess_data.py."
        )


# ------------------------------------------------------------------
# Nested CV (shared by main and sensitivity analysis)
# ------------------------------------------------------------------

def run_nested_cv(data_path, output_dir, pipelines, param_grids,
                  conditions=("SD", "ID"),
                  outer_splits=5, inner_splits=3, inner_repeats=2,
                  random_state=42, block_size_by_condition=None,
                  extra_summary_fields=None, file_prefix="participant"):
    """
    Participant-wise nested cross-validation.

    Outer loop: stratified K-fold; the held-out fold is only used for the
    final prediction. Inner loop: repeated stratified K-fold grid search over
    all candidate pipelines; the pipeline with the best inner-CV accuracy is
    refit on the outer training set and used to predict the outer test fold.

    Parameters
    ----------
    data_path : folder containing <participant>/<participant>_<cond>_X.npy / _y.npy
    output_dir : where the result CSVs are written
    pipelines, param_grids : dicts from build_pipelines_*()
    block_size_by_condition : {condition: channels per chromophore}; only
        needed for block_covariances pipelines
    extra_summary_fields : dict added to every summary row (e.g. latent_dim)

    Writes <file_prefix>_summary_<timestamp>.csv (one row per participant and
    condition) and <file_prefix>_trial_predictions_<timestamp>.csv (one row
    per trial). Returns (summary_csv_path, trials_csv_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    participants = sorted(p for p in os.listdir(data_path) if p.startswith("P"))
    summary_rows, trial_rows = [], []

    for participant in tqdm(participants, desc="Participants"):
        for condition in conditions:
            X_path = os.path.join(data_path, participant, f"{participant}_{condition}_X.npy")
            y_path = os.path.join(data_path, participant, f"{participant}_{condition}_y.npy")
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                continue

            X, y = np.load(X_path), np.load(y_path)
            if block_size_by_condition:
                check_block_size(X, block_size_by_condition.get(condition), f"{participant} {condition}: ")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
            inner_cv = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=inner_repeats,
                                               random_state=random_state)

            y_pred = np.empty_like(y)
            fold_idx_all = np.empty_like(y, dtype=int)
            chosen_pipelines = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                best_score, best_model, best_name, best_params = -np.inf, None, None, None
                for pipe_name, pipe in pipelines.items():
                    pipe_instance = copy.deepcopy(pipe)
                    if block_size_by_condition and "block_covariances" in pipe_name:
                        bs = block_size_by_condition.get(condition)
                        if bs is not None:
                            pipe_instance.set_params(block_covariances_cov__block_size=bs)

                    gs = GridSearchCV(pipe_instance, param_grids[pipe_name], cv=inner_cv,
                                      scoring="accuracy", n_jobs=-1)
                    gs.fit(X_train, y_train)
                    if gs.best_score_ > best_score:
                        best_score = gs.best_score_
                        best_model = copy.deepcopy(gs.best_estimator_)
                        best_name = pipe_name
                        best_params = gs.best_params_

                best_model.fit(X_train, y_train)
                fold_preds = best_model.predict(X_test)
                y_pred[test_idx] = fold_preds
                fold_idx_all[test_idx] = fold_idx
                chosen_pipelines.append({
                    "fold": fold_idx, "pipeline": best_name, "params": best_params,
                    "fold_accuracy": float(accuracy_score(y_test, fold_preds)),
                })

            row = {
                "participant": participant, "condition": condition,
                "accuracy": float(accuracy_score(y, y_pred)), "outer_folds": outer_splits,
                "n_trials": int(len(y)), "chosen_models": json.dumps(chosen_pipelines),
            }
            if extra_summary_fields:
                row.update(extra_summary_fields)
            summary_rows.append(row)

            for i in range(len(y)):
                trial_rows.append({
                    "participant": participant, "condition": condition,
                    "trial_idx": int(i), "fold_idx": int(fold_idx_all[i]),
                    "y_true": y[i], "y_pred": y_pred[i],
                })

    summary_csv = os.path.join(output_dir, f"{file_prefix}_summary_{timestamp}.csv")
    trials_csv = os.path.join(output_dir, f"{file_prefix}_trial_predictions_{timestamp}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(trial_rows).to_csv(trials_csv, index=False)
    return summary_csv, trials_csv


# ------------------------------------------------------------------
# Restricted permutation test (used for "vs. chance" tests)
# ------------------------------------------------------------------
# Unlike a post-hoc label shuffle (shuffle labels, compare with fixed
# predictions), the pipeline is actually REFIT for every permutation
# (Ojala & Garriga, 2010). To keep this computationally feasible, no
# grid search is repeated (default hyperparameters are used). The two
# degrees of freedom of the pipeline architecture are treated differently:
#
#   - Feature-extraction method (xdawn / covariances / block_covariances):
#     fixed once per participant and condition to the method chosen most
#     often across the outer folds of the real nested-CV analysis, and kept
#     constant for the observed data and every permutation.
#   - Classifier (svc / tangent_logreg / mdm): re-selected for the observed
#     data AND for every permutation. The winner is the classifier with the
#     highest pooled out-of-fold accuracy for the fixed feature method.
#
# The selection step is therefore part of the permutation loop and affects
# the observed value and the null distribution equally. Runtime scales with
# the number of classifier candidates (up to 3 fits per permutation).

CLASSIFIER_SUFFIXES = ("svc", "tangent_logreg", "mdm")


def _split_pipeline_name(pipeline_name):
    """Split e.g. 'block_covariances_tangent_logreg' into ('block_covariances', 'tangent_logreg')."""
    for clf in CLASSIFIER_SUFFIXES:
        suffix = f"_{clf}"
        if pipeline_name.endswith(suffix):
            return pipeline_name[: -len(suffix)], clf
    raise ValueError(f"Could not extract a classifier suffix from '{pipeline_name}'")


def refit_fixed_pipeline_cv(X, y, pipeline, outer_splits=5, random_state=42):
    """Cross-validate ONE fixed pipeline (no hyperparameter search); return pooled out-of-fold accuracy."""
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    y_pred = np.empty_like(y)
    for train_idx, test_idx in outer_cv.split(X, y):
        model = copy.deepcopy(pipeline)
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return accuracy_score(y, y_pred)


def _prepare_fixed_pipeline(pipeline_template, block_size=None):
    """Deep-copy a pipeline template and set block_size if it has a block_covariances step."""
    pipe = copy.deepcopy(pipeline_template)
    if block_size is not None:
        try:
            pipe.set_params(block_covariances_cov__block_size=block_size)
        except ValueError:
            pass  # not a block_covariances pipeline
    return pipe


def refit_best_classifier_cv(X, y, fe_method, pipelines_lookup, outer_splits=5,
                             random_state=42, block_size=None,
                             classifiers=CLASSIFIER_SUFFIXES):
    """
    For a fixed feature-extraction method, evaluate all classifier candidates
    by K-fold CV (default hyperparameters) and return the best pooled
    out-of-fold accuracy together with the winning classifier. Called
    identically for the observed labels and for every permutation.

    Candidates missing from pipelines_lookup are skipped (the sensitivity
    mode only defines xDAWN pipelines). Returns (best_accuracy, classifier_name).
    """
    if "block_covariances" in fe_method:
        check_block_size(X, block_size, f"[{fe_method}] ")
    best_acc, best_clf = -np.inf, None
    for clf_suffix in classifiers:
        pipe_name = f"{fe_method}_{clf_suffix}"
        if pipe_name not in pipelines_lookup:
            continue
        pipe = _prepare_fixed_pipeline(pipelines_lookup[pipe_name], block_size)
        acc = refit_fixed_pipeline_cv(X, y, pipe, outer_splits, random_state)
        if acc > best_acc:
            best_acc, best_clf = acc, clf_suffix

    if best_clf is None:
        raise ValueError(f"No pipeline for feature method '{fe_method}' found in pipelines_lookup "
                         f"(expected e.g. '{fe_method}_svc').")
    return best_acc, best_clf


def restricted_permutation_null_accuracies(participant_data, fe_method_by_participant, pipelines_lookup,
                                           n_perm, rng, outer_splits=5, random_state=42, block_size=None,
                                           classifiers=CLASSIFIER_SUFFIXES):
    """
    Core of the restricted permutation test. For each participant the feature
    method is fixed, while the classifier is re-selected on every call: once
    on the true labels (observed) and n_perm times on labels shuffled within
    the participant (null distribution). Group- and individual-level p-values
    can both be derived from the result (see core/stats_core.py).

    Parameters
    ----------
    participant_data : {participant: (X, y)}
    fe_method_by_participant : {participant: feature method name}
        (see modal_fe_method_per_participant)
    pipelines_lookup : {pipeline_name: pipeline template}

    Returns
    -------
    participants : list
    observed_accs : {participant: accuracy}
    null_matrix : ndarray, shape (n_perm, n_participants)
    observed_classifier : {participant: classifier that won on the true labels}
    """
    participants = list(participant_data.keys())

    observed_accs, observed_classifier = {}, {}
    for p in participants:
        X, y = participant_data[p]
        acc, clf = refit_best_classifier_cv(
            X, y, fe_method_by_participant[p], pipelines_lookup,
            outer_splits, random_state, block_size, classifiers)
        observed_accs[p] = acc
        observed_classifier[p] = clf

    null_matrix = np.empty((n_perm, len(participants)))
    for i in tqdm(range(n_perm), desc="Restricted permutations"):
        for j, p in enumerate(participants):
            X, y = participant_data[p]
            acc, _ = refit_best_classifier_cv(
                X, rng.permutation(y), fe_method_by_participant[p], pipelines_lookup,
                outer_splits, random_state, block_size, classifiers)
            null_matrix[i, j] = acc

    return participants, observed_accs, null_matrix, observed_classifier


def modal_fe_method_per_participant(df_summary, condition):
    """
    For each participant, return the feature-extraction method (not the full
    pipeline) chosen most often across the outer folds, read from the
    'chosen_models' column written by run_nested_cv. Returns {participant: method}.
    """
    result = {}
    sub = df_summary[df_summary["condition"] == condition]
    for _, row in sub.iterrows():
        folds = json.loads(row["chosen_models"])
        fe_methods = [_split_pipeline_name(f["pipeline"])[0] for f in folds if f.get("pipeline")]
        if not fe_methods:
            continue
        result[row["participant"]] = pd.Series(fe_methods).value_counts().idxmax()
    return result
