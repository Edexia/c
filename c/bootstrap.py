"""Bootstrap confidence intervals for QWK with separate variance components.

Provides 3 distinct variance sources:
- GT (Teacher) noise: Uncertainty in ground truth labels
- Grading (LLM) noise: Instability in the grading system
- Essay sampling: Variance from which essays were sampled
"""

import numpy as np
from numpy.random import Generator

from .metrics import compute_qwk

TeacherNoiseModel = dict[int, float]  # deviation -> probability
StabilityVector = dict[int, float]    # deviation -> probability


def _build_gt_noise_distribution(
    score: int,
    min_grade: int,
    max_grade: int,
    teacher_noise: TeacherNoiseModel,
) -> tuple[list[int], list[float]]:
    """Build probability distribution for GT noise with boundary accumulation."""
    scores = []
    probs = []
    lower_accumulator = 0.0
    upper_accumulator = 0.0
    for deviation, prob in teacher_noise.items():
        new_score = score + deviation
        if new_score < min_grade:
            lower_accumulator += prob
        elif new_score > max_grade:
            upper_accumulator += prob
        else:
            scores.append(new_score)
            probs.append(prob)
    if lower_accumulator > 0:
        if min_grade in scores:
            idx = scores.index(min_grade)
            probs[idx] += lower_accumulator
        else:
            scores.append(min_grade)
            probs.append(lower_accumulator)
    if upper_accumulator > 0:
        if max_grade in scores:
            idx = scores.index(max_grade)
            probs[idx] += upper_accumulator
        else:
            scores.append(max_grade)
            probs.append(upper_accumulator)
    total = sum(probs)
    probs = [p / total for p in probs]
    return scores, probs


def apply_gt_noise(
    score: int,
    rng: Generator,
    teacher_noise: TeacherNoiseModel,
    min_grade: int = 0,
    max_grade: int = 40,
) -> int:
    """Apply ground truth noise model to a score."""
    scores, probs = _build_gt_noise_distribution(score, min_grade, max_grade, teacher_noise)
    return int(rng.choice(scores, p=probs))


def apply_grading_noise(
    score: int, stability_vector: StabilityVector, rng: Generator,
    min_grade: int = 0, max_grade: int = 40
) -> int:
    """Apply grading/LLM stability noise to a score."""
    deviations = list(stability_vector.keys())
    probs = list(stability_vector.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    deviation = int(rng.choice(deviations, p=probs))
    noised_score = score + deviation
    return max(min_grade, min(max_grade, noised_score))


def _compute_ci(qwk_scores: list[float], confidence_level: float) -> tuple[float, float, float]:
    """Compute mean and confidence interval from QWK scores."""
    qwk_array = np.array(qwk_scores)
    qwk_array = qwk_array[~np.isnan(qwk_array)]
    if len(qwk_array) == 0:
        return float('nan'), float('nan'), float('nan')
    mean_qwk = float(np.mean(qwk_array))
    alpha = 1 - confidence_level
    lower_ci = float(np.percentile(qwk_array, 100 * alpha / 2))
    upper_ci = float(np.percentile(qwk_array, 100 * (1 - alpha / 2)))
    return mean_qwk, lower_ci, upper_ci


def bootstrap_qwk_gt_only(
    predicted: list[int],
    ground_truth: list[int],
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY GT/teacher noise.

    Isolates variance from ground truth uncertainty - how much would
    QWK change if the human graders had given slightly different scores?

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []
    for _ in range(n_iterations):
        noised_gt = [apply_gt_noise(gt, rng, teacher_noise, min_grade, max_grade) for gt in ground_truth]
        qwk = compute_qwk(predicted, noised_gt)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_grading_only(
    predicted: list[int],
    ground_truth: list[int],
    stability_vector: StabilityVector,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY grading/LLM noise.

    Isolates variance from grading instability - how much would
    QWK change if we ran the grading again with different noise?

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []
    for _ in range(n_iterations):
        noised_predicted = [
            apply_grading_noise(pred, stability_vector, rng, min_grade, max_grade)
            for pred in predicted
        ]
        qwk = compute_qwk(noised_predicted, ground_truth)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_sampling_only(
    predicted: list[int],
    ground_truth: list[int],
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY essay sampling variance.

    Uses bootstrap resampling to estimate how QWK would vary if we had
    drawn a different sample of essays. No noise is applied.

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(predicted)
    qwk_scores = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        sampled_predicted = [predicted[idx] for idx in indices]
        sampled_gt = [ground_truth[idx] for idx in indices]
        qwk = compute_qwk(sampled_predicted, sampled_gt)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_paired(
    by_idx: dict[int, tuple[list[int], list[int], list[str]]],
    stability_vectors: dict[int, StabilityVector],
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 2000,
    seed: int = 42,
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Compute pairwise P(A > B) using paired bootstrap scenarios.

    For each iteration, applies the SAME scenario to all files:
    1. Generate ONE set of resample indices (shared)
    2. Generate ONE GT noise per essay index (shared)
    3. For each file: apply grading noise using that file's stability vector
    4. Compute QWK for each file and record pairwise winners

    Args:
        by_idx: Dict mapping file index -> (predicted, ground_truth, target_ids)
        stability_vectors: Dict mapping file index -> stability_vector
        teacher_noise: Teacher noise model (deviation -> probability)
        n_iterations: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Tuple of (win_matrix, avg_qwk_by_idx)
        win_matrix: Dict mapping (i, j) -> P(file i has higher QWK than file j)
    """
    rng = np.random.default_rng(seed)
    indices_list = sorted(by_idx.keys())
    n_samples = len(by_idx[indices_list[0]][0])

    all_grades = []
    for idx in indices_list:
        predicted, ground_truth, _ = by_idx[idx]
        all_grades.extend(predicted)
        all_grades.extend(ground_truth)
    min_grade = min(all_grades)
    max_grade = max(all_grades)

    win_counts: dict[tuple[int, int], int] = {}
    valid_comparisons: dict[tuple[int, int], int] = {}
    qwk_sums: dict[int, float] = {idx: 0.0 for idx in indices_list}
    qwk_counts: dict[int, int] = {idx: 0 for idx in indices_list}

    for i in indices_list:
        for j in indices_list:
            win_counts[(i, j)] = 0
            valid_comparisons[(i, j)] = 0

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        gt_noise_cache: dict[int, int] = {}
        qwks: dict[int, float] = {}

        for idx in indices_list:
            predicted, ground_truth, _ = by_idx[idx]
            sv = stability_vectors.get(idx, {0: 1.0})

            noised_predicted = []
            noised_gt = []

            for sample_idx in indices:
                noised_pred = apply_grading_noise(predicted[sample_idx], sv, rng, min_grade, max_grade)
                noised_predicted.append(noised_pred)

                if sample_idx not in gt_noise_cache:
                    gt_noise_cache[sample_idx] = apply_gt_noise(
                        ground_truth[sample_idx], rng, teacher_noise, min_grade, max_grade
                    )
                noised_gt.append(gt_noise_cache[sample_idx])

            qwk = compute_qwk(noised_predicted, noised_gt)
            qwks[idx] = qwk
            if not np.isnan(qwk):
                qwk_sums[idx] += qwk
                qwk_counts[idx] += 1

        for i in indices_list:
            for j in indices_list:
                if not np.isnan(qwks[i]) and not np.isnan(qwks[j]):
                    valid_comparisons[(i, j)] += 1
                    if qwks[i] > qwks[j]:
                        win_counts[(i, j)] += 1

    win_matrix = {
        k: v / valid_comparisons[k] if valid_comparisons[k] > 0 else float('nan')
        for k, v in win_counts.items()
    }
    avg_qwk_by_idx = {
        idx: qwk_sums[idx] / qwk_counts[idx] if qwk_counts[idx] > 0 else float('nan')
        for idx in indices_list
    }

    return win_matrix, avg_qwk_by_idx


def bootstrap_qwk_combined(
    predicted: list[int],
    ground_truth: list[int],
    teacher_noise: TeacherNoiseModel,
    stability_vector: StabilityVector,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ALL noise sources combined.

    Applies all three variance sources simultaneously in each bootstrap iteration:
    1. Resample essays (sampling variance)
    2. Apply GT noise to resampled ground truth
    3. Apply grading noise to resampled predictions

    This gives the total uncertainty incorporating all sources of error.

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(predicted)
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []

    for _ in range(n_iterations):
        # 1. Resample essays
        indices = rng.integers(0, n_samples, size=n_samples)
        sampled_predicted = [predicted[idx] for idx in indices]
        sampled_gt = [ground_truth[idx] for idx in indices]

        # 2. Apply GT noise to resampled ground truth
        noised_gt = [
            apply_gt_noise(gt, rng, teacher_noise, min_grade, max_grade)
            for gt in sampled_gt
        ]

        # 3. Apply grading noise to resampled predictions
        noised_predicted = [
            apply_grading_noise(pred, stability_vector, rng, min_grade, max_grade)
            for pred in sampled_predicted
        ]

        qwk = compute_qwk(noised_predicted, noised_gt)
        qwk_scores.append(qwk)

    return _compute_ci(qwk_scores, confidence_level)


def get_default_stability_vector() -> StabilityVector:
    """Get a default grading stability vector."""
    return {
        -1: 0.05,
        0: 0.90,
        1: 0.05,
    }
