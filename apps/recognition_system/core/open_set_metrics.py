"""
Open-Set Recognition Evaluation Metrics

This module implements evaluation metrics specific to open-set face recognition,
where the test set contains both known identities (in the gallery) and unknown
identities (not in the gallery).

Key Metrics:
- OSR (Open-Set Recognition Rate): Overall accuracy including unknowns
- KCA (Known Class Accuracy): Closed-set accuracy on known samples
- UDR (Unknown Detection Rate): Recall for unknown class
- Precision/Recall for unknown class
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class OpenSetMetrics:
    """
    Open-set recognition evaluation metrics.

    Confusion Matrix for Open-Set:
    ┌────────────┬─────────────┬──────────────┐
    │            │ Pred Known  │ Pred Unknown │
    ├────────────┼─────────────┼──────────────┤
    │ GT Known   │ TP_known    │ FN_known     │
    │ GT Unknown │ FP_unknown  │ TN_unknown   │
    └────────────┴─────────────┴──────────────┘

    Attributes:
        true_known_correct: Known correctly identified (TP for known)
        true_known_incorrect: Known misidentified as different identity (FP for known, wrong ID)
        true_known_as_unknown: Known rejected as unknown (FN for known)
        true_unknown_as_known: Unknown accepted as known (FP for unknown)
        true_unknown_correct: Unknown correctly rejected (TN for unknown)
        osr: Open-Set Recognition Rate
        kca: Known Class Accuracy
        udr: Unknown Detection Rate
        precision_unknown: Precision for unknown class
        recall_unknown: Recall for unknown class (same as UDR)
        f1_unknown: F1-score for unknown class
        overall_accuracy: Overall accuracy (TP + TN) / Total
    """

    # Confusion matrix components
    true_known_correct: int = 0
    true_known_incorrect: int = 0
    true_known_as_unknown: int = 0
    true_unknown_as_known: int = 0
    true_unknown_correct: int = 0

    # Derived metrics
    osr: float = 0.0
    kca: float = 0.0
    udr: float = 0.0
    precision_unknown: float = 0.0
    recall_unknown: float = 0.0
    f1_unknown: float = 0.0
    overall_accuracy: float = 0.0

    # Per-identity metrics
    per_identity_correct: Dict[str, int] = None
    per_identity_total: Dict[str, int] = None
    per_identity_accuracy: Dict[str, float] = None


def compute_open_set_metrics(
    predictions: List[Dict], ground_truth: List[Dict], known_labels: List[str]
) -> OpenSetMetrics:
    """
    Compute open-set recognition metrics.

    Args:
        predictions: List of prediction dicts with keys:
            - 'name': predicted identity or "Unknown"
            - 'accepted': whether match was accepted (bool)
        ground_truth: List of ground truth dicts with keys:
            - 'name': true identity
            - 'is_unknown': whether this is an unknown sample (bool)
        known_labels: List of known identity names (gallery members)

    Returns:
        OpenSetMetrics object with computed metrics

    Formulas:
        OSR = (true_known_correct + true_unknown_correct) / total_samples
        KCA = true_known_correct / total_known_samples
        UDR = true_unknown_correct / total_unknown_samples
        Precision_unknown = true_unknown_correct / (true_unknown_correct + true_known_as_unknown)
        Recall_unknown = true_unknown_correct / total_unknown_samples (same as UDR)
        F1_unknown = 2 * (Precision * Recall) / (Precision + Recall)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions and ground truth must have same length "
            f"({len(predictions)} vs {len(ground_truth)})"
        )

    metrics = OpenSetMetrics()
    metrics.per_identity_correct = {}
    metrics.per_identity_total = {}
    metrics.per_identity_accuracy = {}

    known_set = set(known_labels)

    for pred, gt in zip(predictions, ground_truth):
        pred_name = pred.get("name", "Unknown")
        pred_accepted = pred.get("accepted", False)
        gt_name = gt.get("name")
        is_unknown = gt.get("is_unknown", False)

        # Track per-identity stats
        if not is_unknown:
            metrics.per_identity_total[gt_name] = (
                metrics.per_identity_total.get(gt_name, 0) + 1
            )

        # Ground truth: Known identity
        if not is_unknown and gt_name in known_set:
            if pred_accepted and pred_name == gt_name:
                # Correctly identified
                metrics.true_known_correct += 1
                metrics.per_identity_correct[gt_name] = (
                    metrics.per_identity_correct.get(gt_name, 0) + 1
                )
            elif pred_accepted and pred_name != gt_name:
                # Misidentified (wrong identity)
                metrics.true_known_incorrect += 1
            elif not pred_accepted:
                # Rejected as unknown (false rejection)
                metrics.true_known_as_unknown += 1

        # Ground truth: Unknown identity
        elif is_unknown or gt_name not in known_set:
            if pred_accepted:
                # Unknown accepted as known (false accept)
                metrics.true_unknown_as_known += 1
            else:
                # Unknown correctly rejected
                metrics.true_unknown_correct += 1

    # Compute aggregate metrics
    total_samples = len(predictions)
    total_known = metrics.true_known_correct + metrics.true_known_incorrect + metrics.true_known_as_unknown
    total_unknown = metrics.true_unknown_as_known + metrics.true_unknown_correct

    # Open-Set Recognition Rate
    if total_samples > 0:
        metrics.osr = (
            metrics.true_known_correct + metrics.true_unknown_correct
        ) / total_samples
        metrics.overall_accuracy = metrics.osr  # Same as OSR

    # Known Class Accuracy (closed-set accuracy on known samples)
    if total_known > 0:
        metrics.kca = metrics.true_known_correct / total_known

    # Unknown Detection Rate (recall for unknown class)
    if total_unknown > 0:
        metrics.udr = metrics.true_unknown_correct / total_unknown
        metrics.recall_unknown = metrics.udr

    # Precision for unknown class
    total_predicted_unknown = metrics.true_unknown_correct + metrics.true_known_as_unknown
    if total_predicted_unknown > 0:
        metrics.precision_unknown = (
            metrics.true_unknown_correct / total_predicted_unknown
        )

    # F1-score for unknown class
    if metrics.precision_unknown > 0 and metrics.recall_unknown > 0:
        metrics.f1_unknown = (
            2
            * metrics.precision_unknown
            * metrics.recall_unknown
            / (metrics.precision_unknown + metrics.recall_unknown)
        )

    # Per-identity accuracy
    for identity in metrics.per_identity_total:
        correct = metrics.per_identity_correct.get(identity, 0)
        total = metrics.per_identity_total[identity]
        metrics.per_identity_accuracy[identity] = correct / total if total > 0 else 0.0

    return metrics


def print_open_set_metrics(metrics: OpenSetMetrics) -> None:
    """
    Print open-set metrics in a formatted table.

    Args:
        metrics: OpenSetMetrics object to print
    """
    print("\n" + "=" * 80)
    print("Open-Set Recognition Metrics")
    print("=" * 80)

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("┌────────────────┬──────────────┬───────────────┐")
    print("│                │ Pred Known   │ Pred Unknown  │")
    print("├────────────────┼──────────────┼───────────────┤")
    print(
        f"│ GT Known       │ {metrics.true_known_correct:12d} │ {metrics.true_known_as_unknown:13d} │"
    )
    print(
        f"│ GT Unknown     │ {metrics.true_unknown_as_known:12d} │ {metrics.true_unknown_correct:13d} │"
    )
    print("└────────────────┴──────────────┴───────────────┘")
    print(
        f"Known Misidentified (wrong ID): {metrics.true_known_incorrect}"
    )

    # Key metrics
    print("\nKey Metrics:")
    print(f"  OSR (Open-Set Recognition Rate): {metrics.osr * 100:.2f}%")
    print(f"  KCA (Known Class Accuracy):      {metrics.kca * 100:.2f}%")
    print(f"  UDR (Unknown Detection Rate):    {metrics.udr * 100:.2f}%")
    print(f"  Precision (Unknown):             {metrics.precision_unknown * 100:.2f}%")
    print(f"  Recall (Unknown):                {metrics.recall_unknown * 100:.2f}%")
    print(f"  F1-Score (Unknown):              {metrics.f1_unknown * 100:.2f}%")

    # Per-identity accuracy (top 10 and bottom 10)
    if metrics.per_identity_accuracy:
        print("\nPer-Identity Accuracy (Top 10):")
        sorted_identities = sorted(
            metrics.per_identity_accuracy.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for identity, accuracy in sorted_identities[:10]:
            total = metrics.per_identity_total.get(identity, 0)
            correct = metrics.per_identity_correct.get(identity, 0)
            print(f"  {identity:20s}: {accuracy * 100:5.1f}% ({correct}/{total})")

        if len(sorted_identities) > 10:
            print("\nPer-Identity Accuracy (Bottom 10):")
            for identity, accuracy in sorted_identities[-10:]:
                total = metrics.per_identity_total.get(identity, 0)
                correct = metrics.per_identity_correct.get(identity, 0)
                print(
                    f"  {identity:20s}: {accuracy * 100:5.1f}% ({correct}/{total})"
                )

    print("=" * 80 + "\n")


def metrics_to_dict(metrics: OpenSetMetrics) -> Dict:
    """
    Convert OpenSetMetrics to dictionary for JSON serialization.

    Args:
        metrics: OpenSetMetrics object

    Returns:
        Dictionary representation
    """
    return {
        "confusion_matrix": {
            "true_known_correct": metrics.true_known_correct,
            "true_known_incorrect": metrics.true_known_incorrect,
            "true_known_as_unknown": metrics.true_known_as_unknown,
            "true_unknown_as_known": metrics.true_unknown_as_known,
            "true_unknown_correct": metrics.true_unknown_correct,
        },
        "osr": metrics.osr,
        "kca": metrics.kca,
        "udr": metrics.udr,
        "precision_unknown": metrics.precision_unknown,
        "recall_unknown": metrics.recall_unknown,
        "f1_unknown": metrics.f1_unknown,
        "overall_accuracy": metrics.overall_accuracy,
        "per_identity_accuracy": metrics.per_identity_accuracy,
    }


def compute_roc_curve(
    similarity_scores: List[float], is_genuine: List[bool], num_thresholds: int = 100
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute ROC curve for open-set recognition (FAR vs TAR).

    Args:
        similarity_scores: List of similarity scores
        is_genuine: List of boolean flags (True=genuine match, False=impostor)
        num_thresholds: Number of threshold points to evaluate

    Returns:
        Tuple of (thresholds, far_list, tar_list)
        - thresholds: List of threshold values
        - far_list: False Accept Rate at each threshold
        - tar_list: True Accept Rate at each threshold
    """
    scores = np.array(similarity_scores)
    genuine = np.array(is_genuine)

    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    far_list = []
    tar_list = []

    genuine_scores = scores[genuine]
    impostor_scores = scores[~genuine]

    for thresh in thresholds:
        # True Accept Rate: genuine samples above threshold
        tar = np.mean(genuine_scores >= thresh) if len(genuine_scores) > 0 else 0.0

        # False Accept Rate: impostor samples above threshold
        far = np.mean(impostor_scores >= thresh) if len(impostor_scores) > 0 else 0.0

        tar_list.append(tar)
        far_list.append(far)

    return thresholds.tolist(), far_list, tar_list
