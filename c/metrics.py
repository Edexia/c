"""Metrics for evaluating grading quality."""

from sklearn.metrics import cohen_kappa_score


def compute_qwk(predicted: list[int], ground_truth: list[int]) -> float:
    """Compute Quadratic Weighted Kappa between predicted and ground truth."""
    if len(predicted) != len(ground_truth):
        raise ValueError("Predicted and ground truth must have same length")
    if len(predicted) == 0:
        return 0.0
    if len(set(predicted)) == 1 or len(set(ground_truth)) == 1:
        return float('nan')
    try:
        return cohen_kappa_score(ground_truth, predicted, weights="quadratic")
    except (ValueError, ZeroDivisionError):
        return float('nan')


def compute_exact_accuracy(predicted: list[int], ground_truth: list[int]) -> float:
    """Compute exact accuracy (predicted == ground truth)."""
    if len(predicted) == 0:
        return 0.0
    matches = sum(p == gt for p, gt in zip(predicted, ground_truth))
    return matches / len(predicted)


def compute_near_accuracy(predicted: list[int], ground_truth: list[int]) -> float:
    """Compute near accuracy (|predicted - ground truth| <= 1)."""
    if len(predicted) == 0:
        return 0.0
    near_matches = sum(abs(p - gt) <= 1 for p, gt in zip(predicted, ground_truth))
    return near_matches / len(predicted)
