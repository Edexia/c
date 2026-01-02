"""Bootstrap confidence intervals with GT and stability noise models.

Teacher noise and LLM stability noise models should be provided by the caller
based on the source EDF file and Experiment 1 data respectively.
"""

import numpy as np
from numpy.random import Generator

from .metrics import compute_qwk

# Type alias for noise models
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


def apply_stability_noise(
    score: int, stability_vector: dict[int, float], rng: Generator,
    min_grade: int = 0, max_grade: int = 40
) -> int:
    """Apply diparative stability noise to a score."""
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
    # Filter out NaN values (from degenerate bootstrap samples)
    qwk_array = qwk_array[~np.isnan(qwk_array)]
    if len(qwk_array) == 0:
        return float('nan'), float('nan'), float('nan')
    mean_qwk = float(np.mean(qwk_array))
    alpha = 1 - confidence_level
    lower_ci = float(np.percentile(qwk_array, 100 * alpha / 2))
    upper_ci = float(np.percentile(qwk_array, 100 * (1 - alpha / 2)))
    return mean_qwk, lower_ci, upper_ci


def bootstrap_qwk(
    predicted: list[int],
    ground_truth: list[int],
    stability_vector: StabilityVector,
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ALL noise sources: resampling + LLM + teacher noise.

    Incorporates:
    - Sampling uncertainty (bootstrap resampling)
    - LLM stability noise (from stability_vector)
    - Teacher/GT noise (from teacher_noise)

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(predicted)
    # Infer grade bounds from data
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        noised_predicted = []
        noised_gt = []
        for idx in indices:
            noised_pred = apply_stability_noise(predicted[idx], stability_vector, rng, min_grade, max_grade)
            noised_predicted.append(noised_pred)
            noised_gt_score = apply_gt_noise(ground_truth[idx], rng, teacher_noise, min_grade, max_grade)
            noised_gt.append(noised_gt_score)
        qwk = compute_qwk(noised_predicted, noised_gt)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_sampling_only(
    predicted: list[int],
    ground_truth: list[int],
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY sampling uncertainty (no noise models).

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


def bootstrap_qwk_llm_only(
    predicted: list[int],
    ground_truth: list[int],
    stability_vector: dict[int, float],
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY LLM stability noise (no resampling, no GT noise).

    Applies LLM stability noise to the FULL dataset each iteration.
    Isolates the uncertainty contribution from LLM instability.

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    # Infer grade bounds from data
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []
    for _ in range(n_iterations):
        noised_predicted = [
            apply_stability_noise(pred, stability_vector, rng, min_grade, max_grade) for pred in predicted
        ]
        qwk = compute_qwk(noised_predicted, ground_truth)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_teacher_only(
    predicted: list[int],
    ground_truth: list[int],
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute QWK CI with ONLY teacher/GT noise (no resampling, no LLM noise).

    Applies teacher noise to the FULL dataset each iteration.
    Isolates the uncertainty contribution from ground truth variability.

    Returns:
        Tuple of (mean_qwk, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    # Infer grade bounds from data
    all_grades = predicted + ground_truth
    min_grade = min(all_grades)
    max_grade = max(all_grades)
    qwk_scores = []
    for _ in range(n_iterations):
        noised_gt = [apply_gt_noise(gt, rng, teacher_noise, min_grade, max_grade) for gt in ground_truth]
        qwk = compute_qwk(predicted, noised_gt)
        qwk_scores.append(qwk)
    return _compute_ci(qwk_scores, confidence_level)


def bootstrap_qwk_paired(
    by_n: dict[int, tuple[list[int], list[int], list[str]]],
    stability_vectors: dict[int, StabilityVector],
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 1000,
    seed: int = 42,
) -> tuple[
    dict[tuple[int, int], float],  # win_matrix: P(n_i > n_j)
    dict[int, float],              # avg_qwk_by_n
]:
    """Compute pairwise P(N=X > N=Y) using paired bootstrap scenarios.

    For each iteration, applies the SAME scenario to all N values:
    1. Generate ONE set of resample indices (shared across all N)
    2. Generate ONE GT noise per essay index (shared across all N)
    3. For each N: apply LLM noise using that N's stability vector
    4. Compute QWK for each N and record pairwise winners

    Args:
        by_n: Dict mapping N -> (predicted, ground_truth, target_ids)
        stability_vectors: Dict mapping N -> stability_vector
        teacher_noise: Teacher noise model (deviation -> probability)
        n_iterations: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Tuple of (win_matrix, avg_qwk_by_n)
    """
    rng = np.random.default_rng(seed)
    n_values = sorted(by_n.keys())
    n_samples = len(by_n[n_values[0]][0])  # All N have same number of targets

    # Infer grade bounds from all data
    all_grades = []
    for n in n_values:
        predicted, ground_truth, _ = by_n[n]
        all_grades.extend(predicted)
        all_grades.extend(ground_truth)
    min_grade = min(all_grades)
    max_grade = max(all_grades)

    # Initialize counters
    win_counts: dict[tuple[int, int], int] = {}
    valid_comparisons: dict[tuple[int, int], int] = {}
    qwk_sums: dict[int, float] = {n: 0.0 for n in n_values}
    qwk_counts: dict[int, int] = {n: 0 for n in n_values}

    for n_i in n_values:
        for n_j in n_values:
            win_counts[(n_i, n_j)] = 0
            valid_comparisons[(n_i, n_j)] = 0

    # Main iteration loop
    for _ in range(n_iterations):
        # STEP 1: Generate SHARED resample indices
        indices = rng.integers(0, n_samples, size=n_samples)

        # STEP 2: Pre-compute SHARED GT noise for each unique original essay index
        # Key: original essay index -> noised GT value for this iteration
        gt_noise_cache: dict[int, int] = {}

        # STEP 3: Compute QWK for each N under this scenario
        qwks: dict[int, float] = {}

        for n in n_values:
            predicted, ground_truth, _ = by_n[n]
            sv = stability_vectors.get(n, {0: 1.0})

            noised_predicted = []
            noised_gt = []

            for idx in indices:
                # Apply LLM noise (per-N stability vector)
                noised_pred = apply_stability_noise(predicted[idx], sv, rng, min_grade, max_grade)
                noised_predicted.append(noised_pred)

                # Apply GT noise (shared - generate once per orig_idx per iteration)
                if idx not in gt_noise_cache:
                    gt_noise_cache[idx] = apply_gt_noise(ground_truth[idx], rng, teacher_noise, min_grade, max_grade)
                noised_gt.append(gt_noise_cache[idx])

            qwk = compute_qwk(noised_predicted, noised_gt)
            qwks[n] = qwk
            # Only add to sum if valid (not NaN)
            if not np.isnan(qwk):
                qwk_sums[n] += qwk
                qwk_counts[n] += 1

        # STEP 4: Record pairwise winners (only if both QWKs are valid)
        for n_i in n_values:
            for n_j in n_values:
                if not np.isnan(qwks[n_i]) and not np.isnan(qwks[n_j]):
                    valid_comparisons[(n_i, n_j)] += 1
                    if qwks[n_i] > qwks[n_j]:
                        win_counts[(n_i, n_j)] += 1

    # Convert counts to probabilities
    win_matrix = {
        k: v / valid_comparisons[k] if valid_comparisons[k] > 0 else float('nan')
        for k, v in win_counts.items()
    }
    avg_qwk_by_n = {
        n: qwk_sums[n] / qwk_counts[n] if qwk_counts[n] > 0 else float('nan')
        for n in n_values
    }

    return win_matrix, avg_qwk_by_n


def bootstrap_essay_level_paired(
    by_n: dict[int, tuple[list[int], list[int], list[str]]],
    stability_vectors: dict[int, StabilityVector],
    teacher_noise: TeacherNoiseModel,
    n_iterations: int = 5000,
    seed: int = 42,
) -> dict[tuple[int, int], float]:
    """Compute pairwise P(N=X >= N=Y) using essay-level comparison.

    For each iteration and each essay:
    - A wins or ties if |pred_A - GT| <= |pred_B - GT| (A is equal or closer to GT)

    We count total essay-level wins+ties across all iterations and essays.

    Args:
        by_n: Dict mapping N -> (predicted, ground_truth, target_ids)
        stability_vectors: Dict mapping N -> stability_vector
        teacher_noise: Teacher noise model (deviation -> probability)
        n_iterations: Number of bootstrap iterations
        seed: Random seed

    Returns:
        win_matrix: Dict mapping (n_i, n_j) -> P(n_i is equal or closer to GT than n_j)
    """
    rng = np.random.default_rng(seed)
    n_values = sorted(by_n.keys())
    n_samples = len(by_n[n_values[0]][0])

    # Infer grade bounds from all data
    all_grades = []
    for n in n_values:
        predicted, ground_truth, _ = by_n[n]
        all_grades.extend(predicted)
        all_grades.extend(ground_truth)
    min_grade = min(all_grades)
    max_grade = max(all_grades)

    # Initialize counters for essay-level wins (including ties)
    essay_wins_or_ties: dict[tuple[int, int], int] = {}
    essay_total: dict[tuple[int, int], int] = {}

    for n_i in n_values:
        for n_j in n_values:
            essay_wins_or_ties[(n_i, n_j)] = 0
            essay_total[(n_i, n_j)] = 0

    # Main iteration loop
    for _ in range(n_iterations):
        # Generate SHARED resample indices
        indices = rng.integers(0, n_samples, size=n_samples)

        # Pre-compute SHARED GT noise for each unique original essay index
        gt_noise_cache: dict[int, int] = {}

        # Compute noised predictions for each N
        noised_preds_by_n: dict[int, list[int]] = {}
        noised_gt: list[int] = []

        for n in n_values:
            predicted, ground_truth, _ = by_n[n]
            sv = stability_vectors.get(n, {0: 1.0})

            noised_predicted = []
            if not noised_gt:  # Only compute GT noise once
                for idx in indices:
                    noised_pred = apply_stability_noise(predicted[idx], sv, rng, min_grade, max_grade)
                    noised_predicted.append(noised_pred)
                    if idx not in gt_noise_cache:
                        gt_noise_cache[idx] = apply_gt_noise(ground_truth[idx], rng, teacher_noise, min_grade, max_grade)
                    noised_gt.append(gt_noise_cache[idx])
            else:
                for idx in indices:
                    noised_pred = apply_stability_noise(predicted[idx], sv, rng, min_grade, max_grade)
                    noised_predicted.append(noised_pred)

            noised_preds_by_n[n] = noised_predicted

        # Compare essay-by-essay for all pairs
        for n_i in n_values:
            for n_j in n_values:
                if n_i == n_j:
                    continue
                preds_i = noised_preds_by_n[n_i]
                preds_j = noised_preds_by_n[n_j]

                for k in range(len(noised_gt)):
                    gt = noised_gt[k]
                    dist_i = abs(preds_i[k] - gt)
                    dist_j = abs(preds_j[k] - gt)

                    essay_total[(n_i, n_j)] += 1
                    if dist_i <= dist_j:  # Win OR tie counts
                        essay_wins_or_ties[(n_i, n_j)] += 1

    # Convert counts to probabilities
    win_matrix = {}
    for key in essay_wins_or_ties:
        if essay_total[key] > 0:
            win_matrix[key] = essay_wins_or_ties[key] / essay_total[key]
        else:
            win_matrix[key] = 0.0

    return win_matrix
