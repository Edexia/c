"""Core analysis functions for EGF grading results."""

import base64
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from edf import EDF
from egf import EGF

from .bootstrap import (
    TeacherNoiseModel,
    StabilityVector,
    bootstrap_qwk_gt_only,
    bootstrap_qwk_grading_only,
    bootstrap_qwk_sampling_only,
    bootstrap_qwk_combined,
    bootstrap_qwk_paired,
    get_default_stability_vector,
)
from .metrics import compute_qwk, compute_exact_accuracy, compute_near_accuracy
from .edf_cache import EDFCache


@dataclass
class EGFData:
    """Data extracted from an EGF file. Contains predicted grades only."""
    path: Path
    name: str
    source_hash: str
    source_task_id: str
    max_grade: int
    grades: list[dict[str, Any]]  # Each dict has 'predicted' and 'submission_id' only
    grading_description: Optional[str] = None


@dataclass
class Essay:
    """An essay with predicted grade (from EGF) and ground truth (from EDF)."""
    submission_id: str
    predicted: int
    ground_truth: int


@dataclass
class EDFData:
    """Data extracted from an EDF file for teacher noise modeling."""
    path: Path
    name: str
    content_hash: str
    task_id: str
    max_grade: int
    teacher_noise: dict[str, TeacherNoiseModel]


@dataclass
class QWKResult:
    """QWK result with confidence intervals for 4 variance components."""
    raw_qwk: float
    exact_accuracy: float
    near_accuracy: float
    n_essays: int

    gt_noise_ci: tuple[float, float, float]
    grading_noise_ci: tuple[float, float, float]
    sampling_ci: tuple[float, float, float]
    combined_ci: tuple[float, float, float]


@dataclass
class AnalysisResult:
    """Complete analysis result for a single EGF file."""
    egf_name: str
    egf_path: Path
    edf_name: Optional[str]
    qwk_result: QWKResult
    noise_assumption: str
    grading_description: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing multiple EGF files."""
    egf_names: list[str]
    win_matrix: dict[tuple[int, int], float]
    avg_qwk: dict[int, float]
    n_iterations: int
    n_common_essays: int


@dataclass
class SubmissionDetail:
    """Complete submission data from EDF for modal display."""
    submission_id: str
    student_name: Optional[str]
    student_id: Optional[str]
    essay_markdown: Optional[str]
    ground_truth_grade: Optional[int]
    gt_distribution: Optional[list[float]]
    pdf_base64: Optional[str] = None  # Base64-encoded PDF data
    gt_justification: Optional[str] = None


@dataclass
class LLMCallDetail:
    """LLM call data for modal display."""
    call_id: str
    pass_number: int
    raw_json: dict


@dataclass
class GradeDetail:
    """Complete grade data from EGF for modal display."""
    submission_id: str
    grade: int
    grade_distribution: Optional[list[float]]
    justification: Optional[str]
    llm_calls: list[LLMCallDetail] = field(default_factory=list)


@dataclass
class GradesTableData:
    """Data structure for the interactive grades table."""
    submissions: dict[str, SubmissionDetail]
    egf_grades: dict[str, dict[str, GradeDetail]]
    egf_names: list[str]
    egf_labels: dict[str, str]
    max_grade: int
    noise_assumption: str


@dataclass
class FullAnalysisResult:
    """Complete analysis result for one or more EGF files."""
    individual_results: list[AnalysisResult]
    comparison: Optional[ComparisonResult] = None
    labels: list[str] = field(default_factory=list)
    legend: dict[str, str] = field(default_factory=dict)
    grades_table: Optional[GradesTableData] = None
    summary_markdown: Optional[str] = None


def load_egf_data(egf_path: Path) -> EGFData:
    """Load an EGF file and extract analysis-ready data."""
    name = egf_path.stem if egf_path.is_file() else egf_path.name

    try:
        with EGF.open(egf_path) as egf:
            grades = []
            for grade in egf.grades:
                if grade.grade is not None:
                    grades.append({
                        'predicted': grade.grade,
                        'submission_id': grade.submission_id,
                    })

            source = egf.source
            grading_config = egf.grading_config

            return EGFData(
                path=egf_path,
                name=name,
                source_hash=source.content_hash if source else "",
                source_task_id=source.task_id if source else "",
                max_grade=source.max_grade if source else 40,
                grades=grades,
                grading_description=grading_config.description if grading_config else None,
            )

    except Exception as e:
        raise ValueError(f"Failed to load EGF file {egf_path}: {e}")


def load_edf_teacher_noise(edf_path: Path) -> EDFData:
    """Load an EDF file and extract teacher noise models."""
    name = edf_path.stem if edf_path.is_file() else edf_path.name

    try:
        with EDF.open(edf_path) as edf:
            teacher_noise = extract_teacher_noise_from_edf(edf)

            return EDFData(
                path=edf_path,
                name=name,
                content_hash=edf.content_hash or "",
                task_id=edf.task_id or "",
                max_grade=edf.max_grade,
                teacher_noise=teacher_noise,
            )

    except Exception as e:
        raise ValueError(f"Failed to load EDF file {edf_path}: {e}")


def extract_teacher_noise_from_edf(edf: EDF) -> dict[str, TeacherNoiseModel]:
    """Extract teacher noise models from EDF using the SDK.

    Returns noise models for all 3 assumptions: optimistic, expected, pessimistic.
    """
    deviation_counts: dict[str, dict[int, float]] = {
        "optimistic": defaultdict(float),
        "expected": defaultdict(float),
        "pessimistic": defaultdict(float),
    }
    total_weights: dict[str, float] = {
        "optimistic": 0.0,
        "expected": 0.0,
        "pessimistic": 0.0,
    }

    for sub in edf.submissions:
        grade = sub.grade
        if grade is None:
            continue

        distributions = sub.distributions
        if not distributions:
            continue

        for assumption, dist in [
            ("optimistic", distributions.optimistic),
            ("expected", distributions.expected),
            ("pessimistic", distributions.pessimistic),
        ]:
            if not dist:
                continue
            for possible_grade, prob in enumerate(dist):
                if prob > 0:
                    deviation = possible_grade - grade
                    deviation_counts[assumption][deviation] += prob
                    total_weights[assumption] += prob

    result: dict[str, TeacherNoiseModel] = {}
    for assumption in ["optimistic", "expected", "pessimistic"]:
        if total_weights[assumption] > 0:
            result[assumption] = {
                dev: count / total_weights[assumption]
                for dev, count in sorted(deviation_counts[assumption].items())
            }
        else:
            result[assumption] = {0: 1.0}

    return result


def get_default_teacher_noise() -> dict[str, TeacherNoiseModel]:
    """Get default teacher noise models."""
    return {
        "optimistic": {-1: 0.10, 0: 0.80, 1: 0.10},
        "expected": {-2: 0.05, -1: 0.20, 0: 0.50, 1: 0.20, 2: 0.05},
        "pessimistic": {-3: 0.05, -2: 0.10, -1: 0.20, 0: 0.30, 1: 0.20, 2: 0.10, 3: 0.05},
    }


def analyze_essays(
    essays: list[Essay],
    egf_data: EGFData,
    teacher_noise: TeacherNoiseModel,
    stability_vector: StabilityVector,
    edf_name: Optional[str] = None,
    noise_assumption: str = "expected",
    seed: int = 42,
) -> AnalysisResult:
    """Analyze essays and compute QWK with variance components."""
    if not essays:
        raise ValueError(f"No essays with ground truth for {egf_data.name}")

    predicted = [e.predicted for e in essays]
    ground_truth = [e.ground_truth for e in essays]

    raw_qwk = compute_qwk(predicted, ground_truth)
    exact_acc = compute_exact_accuracy(predicted, ground_truth)
    near_acc = compute_near_accuracy(predicted, ground_truth)

    gt_ci = bootstrap_qwk_gt_only(predicted, ground_truth, teacher_noise, seed=seed)
    grading_ci = bootstrap_qwk_grading_only(predicted, ground_truth, stability_vector, seed=seed)
    sampling_ci = bootstrap_qwk_sampling_only(predicted, ground_truth, seed=seed)
    combined_ci = bootstrap_qwk_combined(
        predicted, ground_truth, teacher_noise, stability_vector, seed=seed
    )

    qwk_result = QWKResult(
        raw_qwk=raw_qwk,
        exact_accuracy=exact_acc,
        near_accuracy=near_acc,
        n_essays=len(essays),
        gt_noise_ci=gt_ci,
        grading_noise_ci=grading_ci,
        sampling_ci=sampling_ci,
        combined_ci=combined_ci,
    )

    return AnalysisResult(
        egf_name=egf_data.name,
        egf_path=egf_data.path,
        edf_name=edf_name,
        qwk_result=qwk_result,
        noise_assumption=noise_assumption,
        grading_description=egf_data.grading_description,
    )


def analyze_multiple_egf(
    egf_data_list: list[EGFData],
    essays_list: list[list[Essay]],
    teacher_noise: TeacherNoiseModel,
    stability_vectors: dict[int, StabilityVector],
    edf_name: Optional[str] = None,
    noise_assumption: str = "expected",
    seed: int = 42,
) -> FullAnalysisResult:
    """Analyze multiple EGF files and optionally compute pairwise comparison."""
    labels = []
    legend = {}
    individual_results = []

    for idx, (egf_data, essays) in enumerate(zip(egf_data_list, essays_list)):
        label = chr(ord('A') + idx) if idx < 26 else f"A{idx - 25}"
        labels.append(label)
        legend[label] = egf_data.name

        sv = stability_vectors.get(idx, get_default_stability_vector())
        result = analyze_essays(
            essays, egf_data, teacher_noise, sv, edf_name, noise_assumption, seed
        )
        individual_results.append(result)

    comparison = None
    if len(egf_data_list) > 1:
        if all_same_source(egf_data_list):
            comparison = compute_comparison(
                essays_list, teacher_noise, stability_vectors, seed
            )
            comparison.egf_names = [egf.name for egf in egf_data_list]

    return FullAnalysisResult(
        individual_results=individual_results,
        comparison=comparison,
        labels=labels,
        legend=legend,
    )


def all_same_source(egf_data_list: list[EGFData]) -> bool:
    """Check if all EGF files reference the same source EDF."""
    if len(egf_data_list) < 2:
        return True

    first_hash = egf_data_list[0].source_hash
    first_task_id = egf_data_list[0].source_task_id

    for egf_data in egf_data_list[1:]:
        if egf_data.source_hash != first_hash or egf_data.source_task_id != first_task_id:
            return False

    return True


def compute_comparison(
    essays_list: list[list[Essay]],
    teacher_noise: TeacherNoiseModel,
    stability_vectors: dict[int, StabilityVector],
    seed: int = 42,
) -> ComparisonResult:
    """Compute pairwise comparison between multiple sets of essays."""
    # Build grade dicts keyed by submission_id
    file_grades: list[dict[str, tuple[int, int]]] = []
    for essays in essays_list:
        grades_dict = {e.submission_id: (e.predicted, e.ground_truth) for e in essays}
        file_grades.append(grades_dict)

    common_ids = set(file_grades[0].keys())
    for grades_dict in file_grades[1:]:
        common_ids &= set(grades_dict.keys())
    common_ids = sorted(common_ids)

    if len(common_ids) == 0:
        raise ValueError("No common essays across all EGF files")

    by_idx: dict[int, tuple[list[int], list[int], list[str]]] = {}
    for idx, grades_dict in enumerate(file_grades):
        predicted = [grades_dict[sid][0] for sid in common_ids]
        ground_truth = [grades_dict[sid][1] for sid in common_ids]
        by_idx[idx] = (predicted, ground_truth, list(common_ids))

    win_matrix, avg_qwk = bootstrap_qwk_paired(
        by_idx, stability_vectors, teacher_noise, seed=seed
    )

    return ComparisonResult(
        egf_names=[],  # Names set by caller
        win_matrix=win_matrix,
        avg_qwk=avg_qwk,
        n_iterations=2000,
        n_common_essays=len(common_ids),
    )


def find_matching_edf(
    egf_data: EGFData,
    edf_cache: EDFCache,
) -> Optional[Path]:
    """Find the EDF file matching an EGF's source hash."""
    return edf_cache.find_by_hash(egf_data.source_hash)


def load_edf_submissions_detail(edf_path: Path, noise_assumption: str = "expected") -> dict[str, SubmissionDetail]:
    """Load detailed submission information from an EDF file for modal display."""
    try:
        with EDF.open(edf_path) as edf:
            submissions = {}
            for sub in edf.submissions:
                gt_distribution = None
                if sub.distributions:
                    dist_map = {
                        'optimistic': sub.distributions.optimistic,
                        'expected': sub.distributions.expected,
                        'pessimistic': sub.distributions.pessimistic,
                    }
                    gt_distribution = dist_map.get(noise_assumption)

                essay_markdown = None
                try:
                    essay_markdown = sub.get_markdown()
                except Exception:
                    pass

                # Load PDF and base64 encode it
                pdf_base64 = None
                try:
                    pdf_bytes = sub.get_pdf()
                    if pdf_bytes:
                        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                except Exception:
                    pass

                gt_justification = None
                if hasattr(sub, 'justification') and sub.justification:
                    gt_justification = sub.justification

                submissions[sub.id] = SubmissionDetail(
                    submission_id=sub.id,
                    student_name=sub.student_name if hasattr(sub, 'student_name') else None,
                    student_id=sub.student_id if hasattr(sub, 'student_id') else None,
                    essay_markdown=essay_markdown,
                    ground_truth_grade=sub.grade,
                    gt_distribution=gt_distribution,
                    pdf_base64=pdf_base64,
                    gt_justification=gt_justification,
                )
            return submissions
    except Exception as e:
        raise ValueError(f"Failed to load EDF submissions from {edf_path}: {e}")


def load_edf_ground_truth(edf_path: Path) -> dict[str, int]:
    """Load ground truth grades from an EDF file, keyed by submission_id."""
    try:
        with EDF.open(edf_path) as edf:
            return {sub.id: sub.grade for sub in edf.submissions if sub.grade is not None}
    except Exception as e:
        raise ValueError(f"Failed to load EDF ground truth from {edf_path}: {e}")


def build_essays(egf_data: EGFData, ground_truth: dict[str, int]) -> list[Essay]:
    """Build Essay list by combining predicted grades from EGF with ground truth from EDF."""
    essays = []
    for g in egf_data.grades:
        sid = g['submission_id']
        gt = ground_truth.get(sid)
        if gt is not None:
            essays.append(Essay(
                submission_id=sid,
                predicted=g['predicted'],
                ground_truth=gt,
            ))
    return essays


def load_egf_grades_detail(egf_path: Path) -> dict[str, GradeDetail]:
    """Load detailed grade information from an EGF file for modal display."""
    try:
        with EGF.open(egf_path) as egf:
            # Group LLM calls by submission
            calls_by_submission: dict[str, list[LLMCallDetail]] = {}
            for call in egf.llm_calls:
                # Extract submission ID and pass number from call_id
                # Format can be:
                #   - custom_p{pass_number}_t{submission_id}  (new format)
                #   - custom_t{submission_id}_p{pass_number}  (old format)
                call_id = call.call_id
                submission_key = None
                pass_num = 0

                if call_id.startswith('custom_p') and '_t' in call_id:
                    # New format: custom_p{pass}_t{submission_id}
                    # e.g., custom_p0_tJoshua_Munivrana_24146573
                    after_custom = call_id[7:]  # Remove 'custom_'
                    parts = after_custom.split('_t', 1)
                    if len(parts) == 2:
                        try:
                            pass_num = int(parts[0][1:])  # Remove 'p' prefix
                        except ValueError:
                            pass_num = 0
                        submission_key = parts[1]
                elif '_p' in call_id:
                    # Old format: custom_t{submission_id}_p{pass_number}
                    parts = call_id.rsplit('_p', 1)
                    if len(parts) == 2:
                        submission_key = parts[0]
                        if submission_key.startswith('custom_t'):
                            submission_key = submission_key[8:]  # len('custom_t') = 8
                        try:
                            pass_num = int(parts[1])
                        except ValueError:
                            pass_num = 0

                if submission_key:
                    if submission_key not in calls_by_submission:
                        calls_by_submission[submission_key] = []

                    calls_by_submission[submission_key].append(LLMCallDetail(
                        call_id=call.call_id,
                        pass_number=pass_num,
                        raw_json=call.to_dict(),
                    ))

            # Sort calls by pass number for each submission
            for submission_key in calls_by_submission:
                calls_by_submission[submission_key].sort(key=lambda x: x.pass_number)

            grades = {}
            for grade in egf.grades:
                if grade.grade is not None:
                    submission_id = grade.submission_id
                    llm_calls = calls_by_submission.get(submission_id, [])

                    grades[submission_id] = GradeDetail(
                        submission_id=submission_id,
                        grade=grade.grade,
                        grade_distribution=grade.grade_distribution if hasattr(grade, 'grade_distribution') else None,
                        justification=grade.justification if hasattr(grade, 'justification') else None,
                        llm_calls=llm_calls,
                    )
            return grades
    except Exception as e:
        raise ValueError(f"Failed to load EGF grades from {egf_path}: {e}")


def build_grades_table_data(
    edf_submissions: dict[str, SubmissionDetail],
    egf_grades_list: list[tuple[str, dict[str, GradeDetail]]],
    labels: list[str],
    max_grade: int,
    noise_assumption: str,
) -> GradesTableData:
    """Build combined grades table data from EDF submissions and EGF grades."""
    egf_grades = {}
    egf_names = []
    egf_labels = {}

    for idx, (egf_name, grades) in enumerate(egf_grades_list):
        egf_names.append(egf_name)
        egf_grades[egf_name] = grades
        label = labels[idx] if idx < len(labels) else chr(ord('A') + idx)
        egf_labels[egf_name] = label

    return GradesTableData(
        submissions=edf_submissions,
        egf_grades=egf_grades,
        egf_names=egf_names,
        egf_labels=egf_labels,
        max_grade=max_grade,
        noise_assumption=noise_assumption,
    )
