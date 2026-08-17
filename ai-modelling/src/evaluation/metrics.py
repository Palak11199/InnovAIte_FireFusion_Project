"""
Shared classification metrics for the ConvLSTM bushfire classifier.

Imported by both the training script and the standalone evaluation script so
that a checkpoint scores identically in both. All functions operate on pooled
1D arrays of labels and probabilities — gridding and masking happen in
flatten_valid() below.

Scores are reported alongside a no-skill baseline, since raw values are not
interpretable at a positive rate near 0.1%.

Argument order follows the sklearn convention (y_true, y_prob) throughout, so
that a mistaken call site is more likely to fail loudly than to return a
plausible wrong number.

Kept dependency-light on purpose (numpy + sklearn only, no torch) so that
importing this module never drags in the training or inference stack.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

EPS = 1e-12


def flatten_valid(y_true, y_prob, valid_mask):
    """
    Flatten gridded labels and predictions to 1D, keeping valid cells only.

    Provided here rather than in each caller so that training and evaluation
    mask identically — an inconsistency here would make their metrics
    incomparable even though the metric functions are shared.

    Inputs:
        y_true (np.ndarray): [n_samples, horizon, height, width, 1] binary labels
        y_prob (np.ndarray): [n_samples, horizon, height, width, 1] probabilities
        valid_mask (np.ndarray): [height, width] boolean, True where cells are valid

    Outputs:
        tuple: (y_true_flat, y_prob_flat), both 1D float arrays over valid cells
    """
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}"
        )

    height, width = valid_mask.shape
    if y_true.shape[2:4] != (height, width):
        raise ValueError(
            f"Grid mismatch: predictions {y_true.shape[2:4]} vs mask {(height, width)}"
        )

    # [N, horizon, H, W, 1] -> [N, horizon, n_valid, 1]
    true_valid = y_true[:, :, valid_mask, :]
    prob_valid = y_prob[:, :, valid_mask, :]

    return true_valid.ravel().astype(np.float64), prob_valid.ravel().astype(np.float64)


def confusion_counts(y_true, y_prob, threshold):
    """
    Count true/false positives and negatives at a given threshold.

    Inputs:
        y_true (np.ndarray): 1D binary labels
        y_prob (np.ndarray): 1D predicted probabilities
        threshold (float): probability at or above which a cell is predicted burning

    Outputs:
        tuple: (tp, fp, fn, tn) as ints
    """
    pred = y_prob >= threshold
    actual = y_true >= 0.5

    return (
        int(np.sum(pred & actual)),
        int(np.sum(pred & ~actual)),
        int(np.sum(~pred & actual)),
        int(np.sum(~pred & ~actual)),
    )


def climatology_baseline(y_true, beta=2.0):
    """
    Reference scores for a model with no skill, computed from labels alone.

    Needed because raw scores are unreadable at this class imbalance: a PR-AUC
    of 0.04 sounds like failure but may be thirty times better than chance.

    Two references are returned:
      - A constant predictor always outputting the base rate. Its PR-AUC is
        exactly the positive rate — the no-skill line on a PR curve.
      - A predictor calling every cell burning. Perfect recall, precision equal
        to the base rate. A trivial rule that a tuned threshold must beat.

    Base rate is computed per split, so validation and test references will
    differ. Do not compare a skill score across splits without checking both.

    Inputs:
        y_true (np.ndarray): 1D binary labels over valid cells
        beta (float): weighting for F-beta, matching compute_metrics()

    Outputs:
        dict: baseline reference values
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    n_total = y_true.size
    base_rate = float((y_true >= 0.5).sum() / n_total) if n_total else 0.0

    b2 = beta ** 2
    always_fbeta = (1 + b2) * base_rate / (b2 * base_rate + 1 + EPS)

    return {
        "base_rate": base_rate,
        "pr_auc_no_skill": base_rate,
        "always_positive_precision": base_rate,
        "always_positive_recall": 1.0,
        f"always_positive_f{beta:g}": float(always_fbeta),
    }


def compute_metrics(y_true, y_prob, threshold, beta=2.0):
    """
    Compute classification metrics on pooled predictions for one split.

    Intended to be called once per split on all predictions at once, not per
    batch and averaged. At this positive rate most batches contain no fire at
    all, so batch-averaged precision and recall are dominated by degenerate
    batches and are not comparable between runs.

    Accuracy is deliberately not reported. At a positive rate near 0.1% a model
    predicting "no fire" everywhere scores above 99.9%, so the number carries no
    information about whether the model works.

    Inputs:
        y_true (np.ndarray): 1D binary labels over valid cells
        y_prob (np.ndarray): 1D predicted probabilities over valid cells
        threshold (float): decision threshold for the thresholded metrics
        beta (float): weighting for F-beta. Above 1 favours recall, on the view
            that a missed fire costs more than a false alarm.

    Outputs:
        dict: metric name -> value. Undefined metrics are None rather than NaN
            so the dict serialises to valid JSON. Keys are stable so that
            training logs and evaluation runs can be compared directly.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}"
        )

    binary = y_true >= 0.5
    n_total = y_true.size
    n_positive = int(binary.sum())
    fbeta_key = f"f{beta:g}"

    tp, fp, fn, tn = confusion_counts(y_true, y_prob, threshold)

    results = {
        "threshold": float(threshold),
        "beta": float(beta),
        "n_cells": n_total,
        "n_positive": n_positive,
        "positive_rate": float(n_positive / n_total) if n_total else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # Calibration: mean squared error of the probabilities themselves.
        # Meaningful for Focal/BCE, which optimise likelihood. Overlap losses
        # such as Tversky do not optimise calibration, so a poor Brier score
        # there is expected and is not evidence the model is worse.
        "brier": float(brier_score_loss(binary, y_prob)) if n_total else None,
        "degenerate": n_positive == 0 or n_positive == n_total,
    }
    results.update(climatology_baseline(y_true, beta=beta))

    if results["degenerate"]:
        # Only one class present. Reporting zeros would be indistinguishable
        # from a model that found nothing, so these are left undefined.
        results.update({
            "precision": None, "recall": None,
            fbeta_key: None, "f1": None,
            "iou": None, "pr_auc": None, "roc_auc": None, "skill": None,
        })
        return results

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    b2 = beta ** 2
    f_beta = (1 + b2) * precision * recall / (b2 * precision + recall + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    # Intersection over union of predicted-burning and actually-burning cells.
    # Ignores the true negatives that dominate the grid.
    iou = tp / (tp + fp + fn + EPS)

    results.update({
        "precision": float(precision),
        "recall": float(recall),
        fbeta_key: float(f_beta),
        "f1": float(f1),
        "iou": float(iou),
        # Threshold-independent. This is the headline number for comparing loss
        # functions, since it does not depend on where the threshold sits.
        "pr_auc": float(average_precision_score(binary, y_prob)),
        # Reported for completeness only. At this positive rate the true
        # negatives dominate the false positive rate, so ROC-AUC looks strong
        # for models that are not useful. Do not compare runs on it.
        "roc_auc": float(roc_auc_score(binary, y_prob)),
    })

    # Ratio of PR-AUC to the no-skill line. 1.0 means the model has learned
    # nothing beyond the base rate. This is the number to quote in a summary,
    # since raw PR-AUC is not interpretable without it.
    base_rate = results["base_rate"]
    results["skill"] = (
        float(results["pr_auc"] / base_rate) if base_rate > 0 else None
    )

    return results


def find_best_threshold(y_true, y_prob, beta=2.0):
    """
    Select the decision threshold maximising F-beta on a validation split.

    A fixed 0.5 is close to meaningless at this class imbalance — the model can
    be well calibrated and still never exceed 0.5 anywhere, predicting no fire
    at all. Overlap losses such as Tversky make this worse, since they do not
    push probabilities towards 0 and 1 at all. The threshold is a property of
    the trained model and belongs in the checkpoint metadata alongside it.

    Should be fitted on validation data only. Choosing it on the test split
    would leak that split into the reported score.

    Inputs:
        y_true (np.ndarray): 1D binary labels over valid cells
        y_prob (np.ndarray): 1D predicted probabilities over valid cells
        beta (float): weighting for F-beta. Above 1 favours recall.

    Outputs:
        tuple: (best_threshold, best_f_beta)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel() >= 0.5
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if not y_true.any() or y_true.all():
        print("WARNING: only one class present — threshold defaulting to 0.5")
        return 0.5, None

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns one more point than thresholds; the final
    # point is the trivial (recall=0, precision=1) endpoint with no threshold.
    precision, recall = precision[:-1], recall[:-1]

    b2 = beta ** 2
    f_beta = (1 + b2) * precision * recall / (b2 * precision + recall + EPS)

    best_idx = int(np.argmax(f_beta))
    return float(thresholds[best_idx]), float(f_beta[best_idx])


def format_metrics(results, title="Metrics"):
    """
    Render a metrics dict as a readable block for logs.

    Inputs:
        results (dict): output of compute_metrics()
        title (str): heading for the block

    Outputs:
        str: formatted multi-line string
    """
    def fmt(key, width=8):
        value = results.get(key)
        return "undefined".ljust(width) if value is None else f"{value:.4f}".ljust(width)

    skill_val = results.get("skill")
    skill = "undefined".ljust(8) if skill_val is None else f"{skill_val:.1f}x".ljust(8)

    lines = [f"\n{title}", "-" * max(len(title), 46)]

    if results.get("degenerate"):
        lines.append("  Only one class present in this split — metrics undefined.")

    beta = results.get("beta", 2.0)
    lines += [
        f"  PR-AUC       {fmt('pr_auc')}  (threshold-independent — compare on this)",
        f"  Skill        {skill}  (vs no-skill PR-AUC of {results['base_rate']:.4f})",
        f"  Threshold    {results['threshold']:.4f}",
        f"  Precision    {fmt('precision')}",
        f"  Recall       {fmt('recall')}",
        f"  F{beta:g}           {fmt(f'f{beta:g}')}",
        f"  F1           {fmt('f1')}",
        f"  IoU          {fmt('iou')}",
        f"  Brier        {fmt('brier')}  (calibration)",
        f"  ROC-AUC      {fmt('roc_auc')}  (inflated by class imbalance)",
        "",
        f"  TP {results['tp']:,}   FP {results['fp']:,}   "
        f"FN {results['fn']:,}   TN {results['tn']:,}",
        f"  Positive rate {results['positive_rate'] * 100:.4f}% "
        f"({results['n_positive']:,} of {results['n_cells']:,} cells)",
        f"  No-skill reference: PR-AUC {results['base_rate']:.4f} | "
        f"always-positive F{beta:g} {results[f'always_positive_f{beta:g}']:.4f}",
    ]
    return "\n".join(lines)
