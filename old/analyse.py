#!/usr/bin/env python3
"""Analyse EGF grading results and generate reports.

Usage:
    cd /path/to/diparative

    # Analyse EGF files directly:
    python 4_analyse_experiment/analyse.py n5.egf n10.egf n20.egf

    # Analyse a folder containing EGF files:
    python 4_analyse_experiment/analyse.py 3_run_experiment/outputs/exp2_dataset_20260102/

    # Auto-detect latest run folder:
    python 4_analyse_experiment/analyse.py --auto

Single file: outputs QWK and accuracy metrics.
Multiple files: outputs NxN comparison matrices + per-file metrics.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

from egf import EGF

# Add the 4_analyse_experiment directory to the path for imports
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from edf import EDF

from lib.bootstrap import (
    bootstrap_qwk,
    bootstrap_qwk_sampling_only,
    bootstrap_qwk_llm_only,
    bootstrap_qwk_teacher_only,
    bootstrap_qwk_paired,
    bootstrap_essay_level_paired,
    TeacherNoiseModel,
    StabilityVector,
)
from lib.metrics import compute_qwk, compute_exact_accuracy, compute_near_accuracy
from lib.report import (
    AnalysisResults,
    StabilityResult,
    ScalingResult,
    ScalingResultByAssumption,
    MultiEGFResults,
    PairwiseComparisonResult,
    save_markdown_report,
    save_json_results,
    save_multi_egf_report,
)

DEFAULT_INPUT_DIR = Path("3_run_experiment/outputs")
DEFAULT_OUTPUT_DIR = Path("4_analyse_experiment/outputs")


# =============================================================================
# EGF File Discovery and Loading
# =============================================================================


def is_egf_archive(path: Path) -> bool:
    """Check if a path is an EGF archive (file or extracted folder).

    EGF archives are either:
    - .egf files (ZIP archives)
    - Folders containing manifest.json (extracted archives)
    """
    if path.is_file():
        # Could be a .egf file or any other extension
        # Try to check if it's a valid ZIP/EGF
        return path.suffix == ".egf" or (path / "manifest.json").exists() == False and path.suffix == ""
    elif path.is_dir():
        # Check for manifest.json (extracted EGF)
        return (path / "manifest.json").exists()
    return False


def resolve_egf_paths(paths: list[str], input_dir: Path) -> list[Path]:
    """Resolve input paths to a list of EGF files/folders.

    Handles:
    - .egf files (ZIP archives)
    - Folders that ARE EGF archives (contain manifest.json)
    - Folders containing EGF files/folders

    Args:
        paths: List of path strings from CLI
        input_dir: Base directory for relative paths

    Returns:
        List of resolved EGF paths (files or folders)
    """
    egf_files: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            # Try relative to current dir first, then input_dir
            if not path.exists():
                path = input_dir / path_str

        if not path.exists():
            print(f"Warning: Path not found: {path_str}")
            continue

        if path.is_file() and path.suffix == ".egf":
            # .egf file
            egf_files.append(path)
        elif path.is_dir():
            # Check if this folder IS an EGF archive (has manifest.json)
            if (path / "manifest.json").exists():
                egf_files.append(path)
            else:
                # Check for .egf files inside
                found_egf = list(path.glob("*.egf"))
                if found_egf:
                    egf_files.extend(found_egf)
                else:
                    # Check for subfolders that are EGF archives
                    found_folders = [p for p in path.iterdir()
                                     if p.is_dir() and (p / "manifest.json").exists()]
                    if found_folders:
                        egf_files.extend(found_folders)
                    else:
                        print(f"Warning: No EGF files/folders found in {path}")

    # Sort by N value extracted from name
    def extract_n(p: Path) -> int:
        # Handle both files (p.stem) and folders (p.name)
        name = p.stem if p.is_file() else p.name
        match = re.search(r"n(\d+)", name)
        return int(match.group(1)) if match else 0

    return sorted(egf_files, key=extract_n)


def find_latest_run_folder(output_dir: Path) -> Optional[Path]:
    """Find the most recent run folder by modification time.

    A run folder contains either:
    - .egf files
    - Subfolders with manifest.json (extracted EGF archives)

    Returns:
        Path to the most recent run folder, or None.
    """
    if not output_dir.exists():
        return None

    run_folders = []
    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        # Check for .egf files
        if list(item.glob("*.egf")):
            run_folders.append(item)
        # Check for EGF subfolders (extracted archives)
        elif any((sub / "manifest.json").exists() for sub in item.iterdir() if sub.is_dir()):
            run_folders.append(item)

    if not run_folders:
        return None

    run_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_folders[0]


def load_egf_data(egf_path: Path) -> dict[str, Any]:
    """Load an EGF file/folder and extract analysis-ready data.

    Args:
        egf_path: Path to .egf file or extracted EGF folder

    Returns:
        Dict with keys: path, n_value, grades (list of dicts with
        'predicted', 'ground_truth', 'target_id'), name
    """
    # Get name from path (works for both files and folders)
    name = egf_path.stem if egf_path.is_file() else egf_path.name

    # Try to extract N from name (e.g., n5.egf, n10, n5)
    n_match = re.search(r"n(\d+)", name)
    n_value = int(n_match.group(1)) if n_match else 0

    grades = []
    with EGF.open(egf_path) as egf:
        for grade in egf.grades:
            if hasattr(grade, 'grade') and grade.grade is not None:
                grades.append({
                    'predicted': grade.grade,
                    'ground_truth': grade.metadata.get('ground_truth'),
                    'target_id': grade.submission_id,
                })
        # Try to get N from first grade's metadata if not in name
        if n_value == 0 and egf.grades:
            first_meta = egf.grades[0].metadata if egf.grades else {}
            n_value = first_meta.get('n', 0)

    return {
        'path': egf_path,
        'n_value': n_value,
        'grades': grades,
        'name': name,
    }


def get_egf_source_info(egf_path: Path) -> Optional[dict[str, Any]]:
    """Extract the source EDF info from an EGF file's manifest.

    Args:
        egf_path: Path to .egf file or extracted EGF folder

    Returns:
        Dict with source EDF info (task_id, content_hash, created_at, max_grade)
        or None if source info could not be extracted
    """
    try:
        with EGF.open(egf_path) as egf:
            source = egf.source
            return {
                'task_id': source.task_id,
                'content_hash': source.content_hash,
                'created_at': source.created_at,
                'max_grade': source.max_grade,
                'egf_path': egf_path,
            }
    except Exception as e:
        print(f"Warning: Could not extract source info from {egf_path}: {e}")
        return None


def validate_egf_sources_match(egf_paths: list[Path]) -> tuple[bool, Optional[str]]:
    """Validate that all EGF files reference the same EDF source.

    For N by N comparisons to be meaningful, all EGF files must have been
    graded against the exact same EDF (same task_id, content_hash, created_at).

    Args:
        egf_paths: List of EGF file paths to validate

    Returns:
        Tuple of (all_match: bool, warning_message: Optional[str])
        - If all match: (True, None)
        - If mismatch: (False, detailed_warning_message)
    """
    if len(egf_paths) < 2:
        return True, None

    # Extract source info from all EGF files
    sources = []
    for path in egf_paths:
        info = get_egf_source_info(path)
        if info is None:
            return False, f"Could not extract source info from {path.name}"
        sources.append(info)

    # Check if all sources match the first one
    reference = sources[0]
    mismatches = []

    for idx, source in enumerate(sources[1:], start=1):
        if (source['task_id'] != reference['task_id'] or
            source['content_hash'] != reference['content_hash'] or
            source['created_at'] != reference['created_at']):
            mismatches.append((egf_paths[idx], source))

    if not mismatches:
        return True, None

    # Build detailed warning message
    def format_timestamp(ts: int) -> str:
        """Format Unix milliseconds timestamp to readable date."""
        from datetime import datetime
        try:
            return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    def format_hash(h: str) -> str:
        """Shorten hash for display."""
        if h and len(h) > 20:
            return h[:20] + "..."
        return h or "N/A"

    lines = [
        "=" * 70,
        "WARNING: N×N comparison is DISABLED - EGF files reference different EDFs",
        "=" * 70,
        "",
        "For pairwise comparisons to be valid, all EGF files must reference",
        "the exact same source EDF. The following files have mismatched sources:",
        "",
        f"Reference ({egf_paths[0].name}):",
        f"  task_id:      {reference['task_id']}",
        f"  content_hash: {format_hash(reference['content_hash'])}",
        f"  created_at:   {format_timestamp(reference['created_at'])}",
        "",
    ]

    for mismatch_path, mismatch_source in mismatches:
        lines.extend([
            f"Mismatch ({mismatch_path.name}):",
            f"  task_id:      {mismatch_source['task_id']}",
            f"  content_hash: {format_hash(mismatch_source['content_hash'])}",
            f"  created_at:   {format_timestamp(mismatch_source['created_at'])}",
            "",
        ])

    lines.extend([
        "The per-file metrics will still be computed, but N×N pairwise",
        "comparison matrices have been skipped.",
        "=" * 70,
    ])

    return False, "\n".join(lines)


def find_egf_files(input_dir: Path, experiment_num: int) -> list[Path]:
    """Find all EGF files for an experiment, sorted by N value.

    Args:
        input_dir: Directory containing EGF files
        experiment_num: Experiment number (1, 2, or 3)

    Returns:
        List of EGF file paths sorted by N value
    """
    pattern = f"exp{experiment_num}_n*_*.egf"
    files = list(input_dir.glob(pattern))

    def extract_n(p: Path) -> int:
        """Extract N value from filename like exp1_n5_dataset_timestamp.egf"""
        match = re.search(r"_n(\d+)_", p.name)
        return int(match.group(1)) if match else 0

    return sorted(files, key=extract_n)


def extract_n_from_path(path: Path) -> int:
    """Extract N value from EGF filename.

    Handles patterns like:
    - exp2_n5_dataset.egf -> 5
    - n5.egf -> 5
    - n10.egf -> 10
    """
    # Try pattern with underscores first (exp2_n5_dataset.egf)
    match = re.search(r"_n(\d+)_", path.name)
    if match:
        return int(match.group(1))
    # Try pattern at start of filename (n5.egf)
    match = re.match(r"n(\d+)\.", path.name)
    if match:
        return int(match.group(1))
    return 0


def parse_submission_id(submission_id: str, experiment_type: str) -> dict[str, Any]:
    """Parse composite submission_id to extract components.

    Args:
        submission_id: Composite ID like "target_s5_p3" or "target_p2"
        experiment_type: One of "permutation_stability", "scaling_n", "content_stability"

    Returns:
        Dict with target_id and experiment-specific fields (set_idx, perm_idx)
    """
    if experiment_type == "permutation_stability":
        # Format: {target_id}_s{set_idx}_p{perm_idx}
        match = re.match(r"(.+)_s(\d+)_p(\d+)$", submission_id)
        if match:
            return {
                "target_id": match.group(1),
                "set_idx": int(match.group(2)),
                "perm_idx": int(match.group(3)),
            }
    elif experiment_type == "scaling_n":
        # Format: {target_id}_p{permutation}
        match = re.match(r"(.+)_p(\d+)$", submission_id)
        if match:
            return {
                "target_id": match.group(1),
                "permutation": int(match.group(2)),
            }
    elif experiment_type == "content_stability":
        # Format: {target_id}_s{set_idx}
        match = re.match(r"(.+)_s(\d+)$", submission_id)
        if match:
            return {
                "target_id": match.group(1),
                "set_idx": int(match.group(2)),
            }

    # Fallback: return as-is
    return {"target_id": submission_id}


def detect_experiment_type(egf_path: Path) -> str:
    """Detect experiment type from EGF filename or content."""
    name = egf_path.name
    if name.startswith("exp1_"):
        return "permutation_stability"
    elif name.startswith("exp2_"):
        return "scaling_n"
    elif name.startswith("exp3_"):
        return "content_stability"
    return "unknown"


def load_egf_as_calls(egf_path: Path) -> list[dict[str, Any]]:
    """Load EGF file and convert grades to call-like dicts for analysis.

    Returns a list of dicts compatible with the existing analysis functions,
    with keys like: n_anchors, target_id, set_idx, perm_idx, extracted_grade, ground_truth
    """
    n_value = extract_n_from_path(egf_path)
    experiment_type = detect_experiment_type(egf_path)

    calls = []
    with EGF.open(egf_path) as egf:
        for grade in egf.grades:
            if hasattr(grade, 'grade') and grade.grade is not None:
                parsed = parse_submission_id(grade.submission_id, experiment_type)
                call = {
                    "n_anchors": n_value,
                    "target_id": parsed.get("target_id"),
                    "extracted_grade": grade.grade,
                    "ground_truth": grade.metadata.get("ground_truth"),
                    "anchor_ids": grade.metadata.get("anchor_ids"),
                }
                # Add experiment-specific fields
                if "set_idx" in parsed:
                    call["set_idx"] = parsed["set_idx"]
                if "perm_idx" in parsed:
                    call["perm_idx"] = parsed["perm_idx"]
                if "permutation" in parsed:
                    call["permutation"] = parsed["permutation"]
                calls.append(call)
    return calls


def load_egf_files_as_data(egf_paths: list[Path]) -> dict[str, Any]:
    """Load multiple EGF files and merge into a single data structure.

    Returns a dict compatible with the existing analysis functions,
    with a 'calls' key containing all grades from all EGF files.
    """
    all_calls = []
    dataset_name = "unknown"
    config = {}

    for path in egf_paths:
        calls = load_egf_as_calls(path)
        all_calls.extend(calls)

        # Extract dataset name from filename
        # Format: exp1_n5_sbc_task_practice_a_20260102.egf
        parts = path.stem.split("_")
        if len(parts) >= 3:
            # Skip exp1, n5, take rest before timestamp
            dataset_parts = parts[2:-1]  # Exclude timestamp
            if dataset_parts:
                dataset_name = "_".join(dataset_parts)

    return {
        "calls": all_calls,
        "config": {"dataset": dataset_name},
    }


def find_latest_egf_experiment_set(input_dir: Path) -> Optional[str]:
    """Find the most recent EGF experiment set by timestamp.

    Returns the timestamp suffix (e.g., "20260102_143022") of the most recent set,
    or None if no EGF files found.
    """
    egf_files = list(input_dir.glob("exp*_n*_*.egf"))
    if not egf_files:
        return None

    # Extract timestamps and find latest
    timestamps = set()
    for f in egf_files:
        # Format: exp1_n5_dataset_20260102_143022.egf
        match = re.search(r"_(\d{8}_\d{6})\.egf$", f.name)
        if match:
            timestamps.add(match.group(1))

    if not timestamps:
        return None

    return max(timestamps)


def find_egf_files_by_timestamp(
    input_dir: Path, experiment_num: int, timestamp: str
) -> list[Path]:
    """Find EGF files for a specific experiment and timestamp."""
    pattern = f"exp{experiment_num}_n*_*_{timestamp}.egf"
    files = list(input_dir.glob(pattern))

    def extract_n(p: Path) -> int:
        match = re.search(r"_n(\d+)_", p.name)
        return int(match.group(1)) if match else 0

    return sorted(files, key=extract_n)


# =============================================================================
# Teacher Noise Model Extraction from EDF
# =============================================================================


def extract_teacher_noise_from_edf(edf_path: Path) -> TeacherNoiseModel:
    """Extract an aggregate teacher noise model from EDF grade distributions.

    Computes the average deviation distribution across all submissions' "expected"
    grade distributions. This represents the typical teacher grading uncertainty.

    Args:
        edf_path: Path to the source EDF file

    Returns:
        Teacher noise model as dict mapping deviation -> probability
    """
    deviation_counts: dict[int, float] = defaultdict(float)
    total_weight = 0.0

    with EDF.open(edf_path) as edf:
        max_grade = edf.max_grade

        for sub in edf.submissions:
            grade = sub.grade  # Ground truth grade
            expected_dist = sub.distributions.expected  # List of probabilities

            # Convert full distribution to deviation-based distribution
            for possible_grade, prob in enumerate(expected_dist):
                if prob > 0:
                    deviation = possible_grade - grade
                    deviation_counts[deviation] += prob
                    total_weight += prob

    # Normalize to get probabilities
    if total_weight > 0:
        return {dev: count / total_weight for dev, count in sorted(deviation_counts.items())}
    else:
        # Fallback: no noise (delta at 0)
        return {0: 1.0}


def extract_all_teacher_noise_from_edf(edf_path: Path) -> dict[str, TeacherNoiseModel]:
    """Extract all 3 teacher noise models from EDF grade distributions.

    Computes the average deviation distribution across all submissions for
    each of the three noise assumptions: optimistic, expected, pessimistic.

    Args:
        edf_path: Path to the source EDF file

    Returns:
        Dict mapping assumption name -> teacher noise model
        Keys: "optimistic", "expected", "pessimistic"
    """
    # Separate deviation counts for each assumption
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

    with EDF.open(edf_path) as edf:
        for sub in edf.submissions:
            grade = sub.grade  # Ground truth grade

            # Process each distribution type
            for assumption, dist in [
                ("optimistic", sub.distributions.optimistic),
                ("expected", sub.distributions.expected),
                ("pessimistic", sub.distributions.pessimistic),
            ]:
                for possible_grade, prob in enumerate(dist):
                    if prob > 0:
                        deviation = possible_grade - grade
                        deviation_counts[assumption][deviation] += prob
                        total_weights[assumption] += prob

    # Normalize to get probabilities for each assumption
    result: dict[str, TeacherNoiseModel] = {}
    for assumption in ["optimistic", "expected", "pessimistic"]:
        if total_weights[assumption] > 0:
            result[assumption] = {
                dev: count / total_weights[assumption]
                for dev, count in sorted(deviation_counts[assumption].items())
            }
        else:
            # Fallback: no noise (delta at 0)
            result[assumption] = {0: 1.0}

    return result


def get_default_teacher_noise() -> TeacherNoiseModel:
    """Get a default teacher noise model if EDF extraction fails.

    This is a reasonable default based on typical marker reliability studies.
    """
    return {
        -2: 0.05,
        -1: 0.20,
        0: 0.50,
        1: 0.20,
        2: 0.05,
    }


def get_default_stability_vector() -> StabilityVector:
    """Get a default stability vector if Exp1 data is unavailable.

    This is a reasonable default assuming mostly stable LLM grading.
    """
    return {
        -1: 0.05,
        0: 0.90,
        1: 0.05,
    }


def find_edf_by_dataset(dataset_name: str) -> Optional[Path]:
    """Find the EDF file for a given dataset name.

    Args:
        dataset_name: Name of the dataset (e.g., "sbc_task_practice_a")

    Returns:
        Path to the EDF file, or None if not found
    """
    data_dir = Path(__file__).parent.parent / "2_clean_data"
    edf_path = data_dir / f"{dataset_name}.edf"
    if edf_path.exists():
        return edf_path
    return None


def extract_teacher_noise_from_egf(egf_paths: list[Path]) -> Optional[TeacherNoiseModel]:
    """Extract teacher noise from EGF files by finding the source EDF.

    Args:
        egf_paths: List of EGF file paths (any one will give us the source EDF reference)

    Returns:
        Teacher noise model, or None if source EDF cannot be found
    """
    if not egf_paths:
        return None

    try:
        with EGF.open(egf_paths[0]) as egf:
            # Get source EDF info from the EGF manifest
            source = egf.source
            task_id = source.task_id

            # Find the EDF file in 2_clean_data
            data_dir = Path(__file__).parent.parent / "2_clean_data"
            for edf_path in data_dir.glob("*.edf"):
                try:
                    with EDF.open(edf_path) as edf:
                        if edf.task_id == task_id:
                            print(f"Extracting teacher noise from: {edf_path.name}")
                            return extract_teacher_noise_from_edf(edf_path)
                except Exception:
                    continue

    except Exception as e:
        print(f"Warning: Could not extract teacher noise from EGF: {e}")

    return None


def extract_all_teacher_noise_from_egf(
    egf_paths: list[Path],
) -> tuple[Optional[dict[str, TeacherNoiseModel]], Optional[Path]]:
    """Extract all 3 teacher noise models from EGF files by finding the source EDF.

    Args:
        egf_paths: List of EGF file paths (any one will give us the source EDF reference)

    Returns:
        Tuple of (noise_models dict, source_edf_path) or (None, None) if not found
        noise_models keys: "optimistic", "expected", "pessimistic"
    """
    if not egf_paths:
        return None, None

    try:
        with EGF.open(egf_paths[0]) as egf:
            # Get source EDF info from the EGF manifest
            source = egf.source
            task_id = source.task_id

            # Find the EDF file in 2_clean_data
            data_dir = Path(__file__).parent.parent / "2_clean_data"
            for edf_path in data_dir.glob("*.edf"):
                try:
                    with EDF.open(edf_path) as edf:
                        if edf.task_id == task_id:
                            print(f"Extracting all teacher noise models from: {edf_path.name}")
                            return extract_all_teacher_noise_from_edf(edf_path), edf_path
                except Exception:
                    continue

    except Exception as e:
        print(f"Warning: Could not extract teacher noise from EGF: {e}")

    return None, None


def get_default_all_teacher_noise() -> dict[str, TeacherNoiseModel]:
    """Get default teacher noise models for all 3 assumptions.

    Returns progressively wider distributions from optimistic to pessimistic.
    """
    return {
        "optimistic": {
            -1: 0.10,
            0: 0.80,
            1: 0.10,
        },
        "expected": {
            -2: 0.05,
            -1: 0.20,
            0: 0.50,
            1: 0.20,
            2: 0.05,
        },
        "pessimistic": {
            -3: 0.05,
            -2: 0.10,
            -1: 0.20,
            0: 0.30,
            1: 0.20,
            2: 0.10,
            3: 0.05,
        },
    }


# =============================================================================
# Legacy JSON Support
# =============================================================================


def resolve_experiment_path(name_or_path: str, input_dir: Path) -> Optional[Path]:
    """Resolve an experiment name or path to a full file path.

    Args:
        name_or_path: Either a full path, a filename, or an experiment name
        input_dir: Directory to search in if name_or_path is just a name

    Returns:
        Path to the experiment file, or None if not found
    """
    path = Path(name_or_path)

    # If it's already a full path that exists, use it
    if path.exists():
        return path

    # If it ends with .json, try looking in input_dir
    if name_or_path.endswith(".json"):
        candidate = input_dir / name_or_path
        if candidate.exists():
            return candidate
        return None

    # Otherwise, try adding .json extension
    candidate = input_dir / f"{name_or_path}.json"
    if candidate.exists():
        return candidate

    return None


def load_experiment_data(path: Path) -> dict[str, Any]:
    """Load experiment output JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyse_experiment1(data: dict[str, Any]) -> list[StabilityResult]:
    """Analyse Experiment 1 (Permutation Stability) data."""
    calls = data.get("calls", [])
    by_n_target_set: dict[int, dict[str, dict[int, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for call in calls:
        n = call.get("n_anchors")
        target_id = call.get("target_id")
        set_idx = call.get("set_idx")
        grade = call.get("extracted_grade")
        if n is not None and target_id and set_idx is not None and grade is not None:
            by_n_target_set[n][target_id][set_idx].append(grade)
    results = []
    for n in sorted(by_n_target_set.keys()):
        all_deviations: dict[int, list[float]] = defaultdict(list)
        num_samples = 0
        for target_id, sets in by_n_target_set[n].items():
            for set_idx, grades in sets.items():
                if not grades:
                    continue
                mode = _compute_mode(grades)
                total = len(grades)
                deviation_counts = defaultdict(int)
                for g in grades:
                    deviation_counts[g - mode] += 1
                for dev, count in deviation_counts.items():
                    all_deviations[dev].append(count / total)
                num_samples += 1
        stability_vector = {}
        standard_errors = {}
        for dev in sorted(all_deviations.keys()):
            props = all_deviations[dev]
            padded = props + [0.0] * (num_samples - len(props))
            stability_vector[dev] = mean(padded)
            standard_errors[dev] = stdev(padded) / (num_samples ** 0.5) if num_samples > 1 else 0.0
        results.append(StabilityResult(
            n_anchors=n,
            stability_vector=stability_vector,
            standard_errors=standard_errors,
            n_samples=num_samples,
        ))
    return results


def _compute_mode(grades: list[int]) -> int:
    """Compute mode of grades (smallest if tie)."""
    counts: dict[int, int] = defaultdict(int)
    for g in grades:
        counts[g] += 1
    max_count = max(counts.values())
    return min(g for g, c in counts.items() if c == max_count)


def analyse_experiment2(
    data: dict[str, Any],
    stability_vectors: dict[int, StabilityVector],
    teacher_noise: TeacherNoiseModel,
    seed: int = 42,
) -> tuple[list[ScalingResult], Optional[PairwiseComparisonResult]]:
    """Analyse Experiment 2 (Scaling N) data."""
    calls = data.get("calls", [])

    # First collect all data by N, keyed by target_id for consistent ordering
    by_n_raw: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    for call in calls:
        n = call.get("n_anchors")
        grade = call.get("extracted_grade")
        gt = call.get("ground_truth")
        target_id = call.get("target_id")
        if n is not None and grade is not None and gt is not None:
            by_n_raw[n][target_id] = (grade, gt)

    # Find target_ids common to ALL N values (for paired comparison)
    all_n_values = list(by_n_raw.keys())
    if all_n_values:
        common_target_ids = set(by_n_raw[all_n_values[0]].keys())
        for n in all_n_values[1:]:
            common_target_ids &= set(by_n_raw[n].keys())
        common_target_ids = sorted(common_target_ids)
    else:
        common_target_ids = []

    # Convert to sorted lists - use ALL valid data for per-N metrics,
    # but track common_target_ids for paired comparison
    by_n: dict[int, tuple[list[int], list[int], list[str]]] = {}
    for n in by_n_raw:
        sorted_ids = sorted(by_n_raw[n].keys())
        pred = [by_n_raw[n][tid][0] for tid in sorted_ids]
        truth = [by_n_raw[n][tid][1] for tid in sorted_ids]
        by_n[n] = (pred, truth, sorted_ids)

    # Separate dict for paired comparison (only common target_ids)
    by_n_paired: dict[int, tuple[list[int], list[int], list[str]]] = {}
    for n in by_n_raw:
        pred = [by_n_raw[n][tid][0] for tid in common_target_ids]
        truth = [by_n_raw[n][tid][1] for tid in common_target_ids]
        by_n_paired[n] = (pred, truth, list(common_target_ids))
    results = []
    for n in sorted(by_n.keys()):
        predicted, ground_truth, target_ids = by_n[n]
        raw_qwk = compute_qwk(predicted, ground_truth)
        exact_acc = compute_exact_accuracy(predicted, ground_truth)
        near_acc = compute_near_accuracy(predicted, ground_truth)
        sv = stability_vectors.get(n, {0: 1.0})

        # Compute all four CI types
        mean_qwk, lower_ci, upper_ci = bootstrap_qwk(
            predicted, ground_truth, sv, teacher_noise, seed=seed
        )
        _, sampling_lower, sampling_upper = bootstrap_qwk_sampling_only(
            predicted, ground_truth, seed=seed
        )
        _, llm_lower, llm_upper = bootstrap_qwk_llm_only(
            predicted, ground_truth, sv, seed=seed
        )
        _, teacher_lower, teacher_upper = bootstrap_qwk_teacher_only(
            predicted, ground_truth, teacher_noise, seed=seed
        )

        results.append(ScalingResult(
            n_anchors=n,
            raw_qwk=raw_qwk,
            mean_qwk=mean_qwk,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            ci_sampling_only_lower=sampling_lower,
            ci_sampling_only_upper=sampling_upper,
            ci_llm_only_lower=llm_lower,
            ci_llm_only_upper=llm_upper,
            ci_teacher_only_lower=teacher_lower,
            ci_teacher_only_upper=teacher_upper,
            exact_accuracy=exact_acc,
            near_accuracy=near_acc,
            n_targets=len(target_ids),
        ))

    # Compute pairwise comparison if we have multiple N values with common targets
    pairwise_result = None
    if len(by_n_paired) > 1 and len(common_target_ids) > 0:
        n_iterations = 5000  # Increased for higher confidence
        print(f"  Paired comparison using {len(common_target_ids)} essays common to all N values")

        # QWK-based comparison
        win_matrix_qwk, avg_qwk = bootstrap_qwk_paired(
            by_n_paired,
            stability_vectors,
            teacher_noise,
            n_iterations=n_iterations,
            seed=seed,
        )

        # Essay-level comparison
        win_matrix_essay = bootstrap_essay_level_paired(
            by_n_paired,
            stability_vectors,
            teacher_noise,
            n_iterations=n_iterations,
            seed=seed,
        )

        pairwise_result = PairwiseComparisonResult(
            n_values=sorted(by_n_paired.keys()),
            win_matrix_qwk=win_matrix_qwk,
            win_matrix_essay=win_matrix_essay,
            n_iterations=n_iterations,
            avg_qwk_by_n=avg_qwk,
        )

    return results, pairwise_result


def analyse_experiment3(data: dict[str, Any]) -> list[StabilityResult]:
    """Analyse Experiment 3 (Content Stability) data.

    Unlike Experiment 1 which groups by (n, target, set) and measures variance across permutations,
    Experiment 3 groups by (n, target) and measures variance across different anchor sets.
    """
    calls = data.get("calls", [])
    by_n_target: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for call in calls:
        n = call.get("n_anchors")
        target_id = call.get("target_id")
        grade = call.get("extracted_grade")
        if n is not None and target_id and grade is not None:
            by_n_target[n][target_id].append(grade)

    results = []
    for n in sorted(by_n_target.keys()):
        all_deviations: dict[int, list[float]] = defaultdict(list)
        num_samples = 0

        for target_id, grades in by_n_target[n].items():
            if not grades:
                continue
            mode = _compute_mode(grades)
            total = len(grades)
            deviation_counts = defaultdict(int)
            for g in grades:
                deviation_counts[g - mode] += 1
            for dev, count in deviation_counts.items():
                all_deviations[dev].append(count / total)
            num_samples += 1

        stability_vector = {}
        standard_errors = {}
        for dev in sorted(all_deviations.keys()):
            props = all_deviations[dev]
            padded = props + [0.0] * (num_samples - len(props))
            stability_vector[dev] = mean(padded)
            standard_errors[dev] = stdev(padded) / (num_samples ** 0.5) if num_samples > 1 else 0.0

        results.append(StabilityResult(
            n_anchors=n,
            stability_vector=stability_vector,
            standard_errors=standard_errors,
            n_samples=num_samples,
        ))

    return results


def find_latest_run_folder(output_dir: Path) -> Optional[Path]:
    """Find the most recent run folder by modification time.

    Run folders are named {dataset}_{timestamp} and contain EGF files.

    Returns:
        Path to the most recent run folder, or None if not found.
    """
    # Look for directories containing EGF files
    run_folders = []
    for item in output_dir.iterdir():
        if item.is_dir() and list(item.glob("exp*.egf")):
            run_folders.append(item)

    if not run_folders:
        return None

    # Sort by modification time, newest first
    run_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_folders[0]


def find_egf_files_in_folder(run_folder: Path) -> list[Path]:
    """Find all EGF files in a run folder.

    Args:
        run_folder: Path to the run folder

    Returns:
        List of EGF file paths sorted by N value.
    """
    def extract_n(p: Path) -> int:
        # Match patterns like n5.egf, n10.egf or exp1_n5.egf
        match = re.search(r"n(\d+)", p.name)
        return int(match.group(1)) if match else 0

    # Find all .egf files
    egf_files = list(run_folder.glob("*.egf"))
    return sorted(egf_files, key=extract_n)


def find_latest_experiments(
    output_dir: Path,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find the most recent experiment outputs by modification time (JSON only - legacy)."""
    exp1_files = list(output_dir.glob("exp1-*.json"))
    exp2_files = list(output_dir.glob("exp2-*.json"))
    exp3_files = list(output_dir.glob("exp3-*.json"))

    # Sort by modification time, newest first
    exp1_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    exp2_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    exp3_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return (
        exp1_files[0] if exp1_files else None,
        exp2_files[0] if exp2_files else None,
        exp3_files[0] if exp3_files else None,
    )


def find_latest_egf_experiments(
    output_dir: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Find the most recent EGF experiment files by timestamp (legacy flat structure).

    Returns:
        Tuple of (exp1_files, exp2_files, exp3_files) - each is a list of
        EGF file paths for different N values, or empty list if not found.
    """
    timestamp = find_latest_egf_experiment_set(output_dir)
    if not timestamp:
        return [], [], []

    exp1_files = find_egf_files_by_timestamp(output_dir, 1, timestamp)
    exp2_files = find_egf_files_by_timestamp(output_dir, 2, timestamp)
    exp3_files = find_egf_files_by_timestamp(output_dir, 3, timestamp)

    return exp1_files, exp2_files, exp3_files


def merge_throughput_stats(
    exp1_data: Optional[dict],
    exp2_data: Optional[dict],
    exp3_data: Optional[dict] = None,
) -> dict[str, Any]:
    """Merge throughput stats from all experiments."""
    stats = {}
    experiments = []

    for exp_name, exp_data in [("exp1", exp1_data), ("exp2", exp2_data), ("exp3", exp3_data)]:
        if exp_data and "throughput_stats" in exp_data:
            experiments.append(exp_name)
            if not stats:
                stats = exp_data["throughput_stats"].copy()
            else:
                for key in ["total_calls", "api_calls", "cached_calls",
                            "total_tokens_in", "total_tokens_out",
                            "total_chars_in", "total_chars_out"]:
                    if key in exp_data["throughput_stats"]:
                        stats[key] = stats.get(key, 0) + exp_data["throughput_stats"][key]
                stats["duration_seconds"] = max(
                    stats.get("duration_seconds", 0),
                    exp_data["throughput_stats"].get("duration_seconds", 0)
                )

    stats["experiment"] = "+".join(experiments) if len(experiments) > 1 else (experiments[0] if experiments else "none")
    return stats


def derive_experiment_id(
    exp1_path: Optional[Path],
    exp2_path: Optional[Path],
    exp3_path: Optional[Path] = None,
) -> str:
    """Derive experiment ID from input experiment names.

    Extracts the adjective-color-animal suffix from experiment names.
    E.g., 'exp1-lunar-purple-goat' -> 'lunar-purple-goat'
    """
    def extract_suffix(path: Path) -> str:
        """Extract the adjective-color-animal suffix from experiment name."""
        stem = path.stem  # e.g., 'exp1-lunar-purple-goat'
        # Remove 'exp1-', 'exp2-', or 'exp3-' prefix
        for prefix in ("exp1-", "exp2-", "exp3-"):
            if stem.startswith(prefix):
                return stem[5:]  # Skip prefix
        return stem

    # Return the first available experiment's suffix
    for path in [exp1_path, exp2_path, exp3_path]:
        if path:
            return extract_suffix(path)
    return "unknown"


def derive_experiment_id_from_egf(egf_paths: list[Path]) -> str:
    """Derive experiment ID from EGF filenames.

    Extracts the timestamp from EGF filename.
    E.g., 'exp1_n5_sbc_task_practice_a_20260102_143022.egf' -> '20260102_143022'
    """
    if not egf_paths:
        return "unknown"

    path = egf_paths[0]
    match = re.search(r"_(\d{8}_\d{6})\.egf$", path.name)
    if match:
        return match.group(1)
    return path.stem


def run_analysis(
    exp1_path: Optional[Path],
    exp2_path: Optional[Path],
    output_dir: Path,
    seed: int = 42,
    exp3_path: Optional[Path] = None,
    teacher_noise_override: Optional[TeacherNoiseModel] = None,
) -> AnalysisResults:
    """Run full analysis on experiment data.

    Args:
        exp1_path: Path to experiment 1 JSON file
        exp2_path: Path to experiment 2 JSON file
        output_dir: Output directory for reports
        seed: Random seed for bootstrap analysis
        exp3_path: Path to experiment 3 JSON file
        teacher_noise_override: Optional override for teacher noise model
    """
    exp1_data = load_experiment_data(exp1_path) if exp1_path else None
    exp2_data = load_experiment_data(exp2_path) if exp2_path else None
    exp3_data = load_experiment_data(exp3_path) if exp3_path else None

    # Find dataset name from any available experiment
    dataset_name = "unknown"
    for data in [exp1_data, exp2_data, exp3_data]:
        if data:
            dataset_name = data.get("config", {}).get("dataset", "unknown")
            break

    # Get teacher noise model
    teacher_noise: TeacherNoiseModel
    teacher_noise_source: str
    if teacher_noise_override is not None:
        teacher_noise = teacher_noise_override
        teacher_noise_source = "CLI override"
    else:
        edf_path = find_edf_by_dataset(dataset_name)
        if edf_path:
            print(f"Extracting teacher noise from EDF: {edf_path.name}")
            teacher_noise = extract_teacher_noise_from_edf(edf_path)
            teacher_noise_source = f"EDF file ({edf_path.name})"
        else:
            print(f"Warning: No EDF found for dataset '{dataset_name}', using default teacher noise")
            teacher_noise = get_default_teacher_noise()
            teacher_noise_source = "Default (no EDF found)"

    stability_results = []
    stability_vectors: dict[int, dict[int, float]] = {}
    llm_noise_source = "Experiment 1 data"

    if exp1_data:
        stability_results = analyse_experiment1(exp1_data)
        for sr in stability_results:
            stability_vectors[sr.n_anchors] = sr.stability_vector

    # Analyse Experiment 3 (content stability)
    content_stability_results = []
    if exp3_data:
        content_stability_results = analyse_experiment3(exp3_data)

    scaling_results = []
    pairwise_results = None
    if exp2_data:
        if not stability_vectors:
            print("Warning: No stability vectors from exp1, using default")
            default_sv = get_default_stability_vector()
            for n in [5, 10, 15, 20, 25, 30]:
                stability_vectors[n] = default_sv.copy()
            llm_noise_source = "Default (no Exp1 data)"
        scaling_results, pairwise_results = analyse_experiment2(
            exp2_data, stability_vectors, teacher_noise, seed
        )

    throughput_stats = merge_throughput_stats(exp1_data, exp2_data, exp3_data)

    results = AnalysisResults(
        stability_results=stability_results,
        scaling_results=scaling_results,
        pairwise_results=pairwise_results,
        dataset_name=dataset_name,
        model_name="gemini-3-flash-preview",
        master_seed=seed,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        throughput_stats=throughput_stats,
        llm_noise_source=llm_noise_source,
        teacher_noise_source=teacher_noise_source,
        content_stability_results=content_stability_results,
    )

    # Create output subdirectory based on experiment ID
    experiment_id = derive_experiment_id(exp1_path, exp2_path, exp3_path)
    experiment_output_dir = output_dir / experiment_id
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    report_path = experiment_output_dir / "report.md"
    json_path = experiment_output_dir / "results.json"

    save_markdown_report(results, report_path)
    save_json_results(results, json_path)

    # Generate histogram images
    from lib.visualizations import save_stability_histograms, save_qwk_chart, save_pairwise_heatmap
    if stability_results:
        save_stability_histograms(stability_results, experiment_output_dir)
    if scaling_results:
        save_qwk_chart(
            scaling_results,
            experiment_output_dir,
            llm_noise_source=llm_noise_source,
            teacher_noise_source=teacher_noise_source,
        )
    if pairwise_results:
        save_pairwise_heatmap(pairwise_results, experiment_output_dir)
    if content_stability_results:
        save_stability_histograms(
            content_stability_results,
            experiment_output_dir,
            filename_prefix="content_stability_histogram",
        )

    print(f"Report saved to {report_path}")
    print(f"JSON results saved to {json_path}")
    print(f"All outputs in: {experiment_output_dir}")

    return results


def run_analysis_from_egf(
    exp1_files: list[Path],
    exp2_files: list[Path],
    exp3_files: list[Path],
    output_dir: Path,
    seed: int = 42,
    teacher_noise_override: Optional[TeacherNoiseModel] = None,
) -> AnalysisResults:
    """Run full analysis from EGF experiment files.

    Args:
        exp1_files: List of EGF files for experiment 1 (one per N value)
        exp2_files: List of EGF files for experiment 2 (one per N value)
        exp3_files: List of EGF files for experiment 3 (one per N value)
        output_dir: Directory to save reports
        seed: Random seed for bootstrap analysis
        teacher_noise_override: Optional override for teacher noise model

    Returns:
        AnalysisResults object with all computed metrics
    """
    # Load EGF files and convert to data format
    exp1_data = load_egf_files_as_data(exp1_files) if exp1_files else None
    exp2_data = load_egf_files_as_data(exp2_files) if exp2_files else None
    exp3_data = load_egf_files_as_data(exp3_files) if exp3_files else None

    # Find dataset name from any available experiment
    dataset_name = "unknown"
    for data in [exp1_data, exp2_data, exp3_data]:
        if data:
            dataset_name = data.get("config", {}).get("dataset", "unknown")
            break

    # Get teacher noise model
    all_files = exp1_files + exp2_files + exp3_files
    teacher_noise: TeacherNoiseModel
    teacher_noise_source: str
    if teacher_noise_override is not None:
        teacher_noise = teacher_noise_override
        teacher_noise_source = "CLI override"
    else:
        # Try to extract from EGF files (via EDF reference)
        teacher_noise_from_egf = extract_teacher_noise_from_egf(all_files)
        if teacher_noise_from_egf:
            teacher_noise = teacher_noise_from_egf
            teacher_noise_source = "Source EDF (via EGF reference)"
        else:
            # Fallback to finding EDF by dataset name
            edf_path = find_edf_by_dataset(dataset_name)
            if edf_path:
                print(f"Extracting teacher noise from EDF: {edf_path.name}")
                teacher_noise = extract_teacher_noise_from_edf(edf_path)
                teacher_noise_source = f"EDF file ({edf_path.name})"
            else:
                print(f"Warning: No EDF found for dataset '{dataset_name}', using default teacher noise")
                teacher_noise = get_default_teacher_noise()
                teacher_noise_source = "Default (no EDF found)"

    stability_results = []
    stability_vectors: dict[int, dict[int, float]] = {}
    llm_noise_source = "Experiment 1 EGF data"

    if exp1_data and exp1_data.get("calls"):
        stability_results = analyse_experiment1(exp1_data)
        for sr in stability_results:
            stability_vectors[sr.n_anchors] = sr.stability_vector

    # Analyse Experiment 3 (content stability)
    content_stability_results = []
    if exp3_data and exp3_data.get("calls"):
        content_stability_results = analyse_experiment3(exp3_data)

    scaling_results = []
    pairwise_results = None
    if exp2_data and exp2_data.get("calls"):
        if not stability_vectors:
            print("Warning: No stability vectors from exp1, using default")
            default_sv = get_default_stability_vector()
            for n in [5, 10, 15, 20, 25, 30]:
                stability_vectors[n] = default_sv.copy()
            llm_noise_source = "Default (no Exp1 data)"
        scaling_results, pairwise_results = analyse_experiment2(
            exp2_data, stability_vectors, teacher_noise, seed
        )

    # No throughput stats from EGF files (they don't store this)
    throughput_stats = {"experiment": "egf_analysis"}

    results = AnalysisResults(
        stability_results=stability_results,
        scaling_results=scaling_results,
        pairwise_results=pairwise_results,
        dataset_name=dataset_name,
        model_name="gemini-3-flash-preview",
        master_seed=seed,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        throughput_stats=throughput_stats,
        llm_noise_source=llm_noise_source,
        teacher_noise_source=teacher_noise_source,
        content_stability_results=content_stability_results,
    )

    # Create output subdirectory based on experiment ID
    experiment_id = derive_experiment_id_from_egf(all_files)
    experiment_output_dir = output_dir / experiment_id
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    report_path = experiment_output_dir / "report.md"
    json_path = experiment_output_dir / "results.json"

    save_markdown_report(results, report_path)
    save_json_results(results, json_path)

    # Generate histogram images
    from lib.visualizations import save_stability_histograms, save_qwk_chart, save_pairwise_heatmap
    if stability_results:
        save_stability_histograms(stability_results, experiment_output_dir)
    if scaling_results:
        save_qwk_chart(
            scaling_results,
            experiment_output_dir,
            llm_noise_source=llm_noise_source,
            teacher_noise_source=teacher_noise_source,
        )
    if pairwise_results:
        save_pairwise_heatmap(pairwise_results, experiment_output_dir)
    if content_stability_results:
        save_stability_histograms(
            content_stability_results,
            experiment_output_dir,
            filename_prefix="content_stability_histogram",
        )

    print(f"Report saved to {report_path}")
    print(f"JSON results saved to {json_path}")
    print(f"All outputs in: {experiment_output_dir}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse EGF grading results and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyse specific EGF files:
    python 4_analyse_experiment/analyse.py n5.egf n10.egf n20.egf

    # Analyse a folder containing EGF files:
    python 4_analyse_experiment/analyse.py 3_run_experiment/outputs/exp2_dataset_20260102/

    # Auto-detect latest run folder:
    python 4_analyse_experiment/analyse.py --auto

Single file: outputs QWK and accuracy metrics.
Multiple files: outputs NxN comparison matrices + per-file metrics.
""",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="EGF files or folders containing EGF files",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect latest run folder",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="3_run_experiment/outputs",
        help="Base directory for --auto (default: 3_run_experiment/outputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="4_analyse_experiment/outputs",
        help="Output directory for reports (default: 4_analyse_experiment/outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for bootstrap analysis (default: 42)",
    )
    parser.add_argument(
        "--teacher-noise",
        type=str,
        default=None,
        help='Override teacher noise model as JSON, e.g., \'{"0": 1.0}\' or \'{"−1": 0.2, "0": 0.6, "1": 0.2}\'',
    )
    return parser.parse_args()


def parse_teacher_noise_json(json_str: str) -> TeacherNoiseModel:
    """Parse teacher noise model from JSON string.

    Args:
        json_str: JSON string like '{"0": 1.0}' or '{"-1": 0.2, "0": 0.6, "1": 0.2}'

    Returns:
        Teacher noise model as dict mapping int deviation -> float probability
    """
    raw = json.loads(json_str)
    return {int(k): float(v) for k, v in raw.items()}


def generate_file_labels(num_files: int) -> list[str]:
    """Generate sequential labels for files (A, B, C, ... Z, AA, AB, ...)."""
    labels = []
    for i in range(num_files):
        if i < 26:
            labels.append(chr(ord('A') + i))
        else:
            # For more than 26 files, use AA, AB, etc.
            labels.append(chr(ord('A') + i // 26 - 1) + chr(ord('A') + i % 26))
    return labels


def analyse_egf_files_with_multi_assumption(
    egf_files: list[Path],
    output_dir: Path,
    all_teacher_noise: dict[str, TeacherNoiseModel],
    stability_vectors: dict[int, StabilityVector],
    seed: int = 42,
    source_edf_name: Optional[str] = None,
    grading_description: Optional[str] = None,
) -> MultiEGFResults:
    """Analyse EGF files with all 3 noise assumptions.

    Computes QWK and CIs for each file under optimistic/expected/pessimistic
    noise assumptions, with 4 CI types (sampling, llm, teacher, combined).

    Args:
        egf_files: List of EGF file paths
        output_dir: Directory to save reports and visualizations
        all_teacher_noise: Dict with keys "optimistic", "expected", "pessimistic"
        stability_vectors: LLM stability vectors by N value
        seed: Random seed for bootstrap
        source_edf_name: Name of source EDF file
        grading_description: Description of grading approach

    Returns:
        MultiEGFResults with all computed metrics
    """
    import numpy as np
    from lib.visualizations import (
        save_pairwise_heatmap,
        save_multi_assumption_qwk_chart,
        save_multi_assumption_grid_chart,
    )

    if not egf_files:
        raise ValueError("No EGF files to analyse")

    # Generate labels for each file (A, B, C, ...)
    labels = generate_file_labels(len(egf_files))
    legend: dict[str, str] = {}

    # Load all EGF files
    print(f"Loading {len(egf_files)} EGF file(s)...")
    file_data = []
    for idx, path in enumerate(egf_files):
        data = load_egf_data(path)
        data['label'] = labels[idx]
        file_data.append(data)
        legend[labels[idx]] = path.name
        print(f"  [{labels[idx]}] {path.name}: {len(data['grades'])} grades")

    # Compute per-file metrics with all assumptions
    print("\nComputing metrics with all noise assumptions...")
    scaling_results: list[ScalingResultByAssumption] = []

    for data in file_data:
        predicted = [g['predicted'] for g in data['grades'] if g['ground_truth'] is not None]
        ground_truth = [g['ground_truth'] for g in data['grades'] if g['ground_truth'] is not None]

        if not predicted:
            print(f"  {data['name']}: No valid grades with ground truth")
            continue

        n_value = data['n_value']
        raw_qwk = compute_qwk(predicted, ground_truth)
        exact_acc = compute_exact_accuracy(predicted, ground_truth)
        near_acc = compute_near_accuracy(predicted, ground_truth)

        print(f"  {data['name']}: Raw QWK={raw_qwk:.4f}")

        # Get stability vector for this N
        sv = stability_vectors.get(n_value, get_default_stability_vector())

        # Compute CIs for each assumption
        by_assumption: dict[str, dict[str, tuple[float, float, float]]] = {}

        for assumption in ["optimistic", "expected", "pessimistic"]:
            teacher_noise = all_teacher_noise[assumption]

            # Sampling only (same for all assumptions)
            mean_sampling, lower_sampling, upper_sampling = bootstrap_qwk_sampling_only(
                predicted, ground_truth, seed=seed
            )

            # LLM only (same for all assumptions - depends on stability vector)
            mean_llm, lower_llm, upper_llm = bootstrap_qwk_llm_only(
                predicted, ground_truth, sv, seed=seed
            )

            # Teacher only (differs by assumption)
            mean_teacher, lower_teacher, upper_teacher = bootstrap_qwk_teacher_only(
                predicted, ground_truth, teacher_noise, seed=seed
            )

            # Combined (differs by assumption)
            mean_combined, lower_combined, upper_combined = bootstrap_qwk(
                predicted, ground_truth, sv, teacher_noise, seed=seed
            )

            by_assumption[assumption] = {
                "sampling_only": (mean_sampling, lower_sampling, upper_sampling),
                "llm_only": (mean_llm, lower_llm, upper_llm),
                "teacher_only": (mean_teacher, lower_teacher, upper_teacher),
                "combined": (mean_combined, lower_combined, upper_combined),
            }

            print(f"    {assumption}: Combined CI [{lower_combined:.3f}, {upper_combined:.3f}]")

        scaling_results.append(ScalingResultByAssumption(
            n_anchors=n_value,
            n_targets=len(predicted),
            raw_qwk=raw_qwk,
            exact_accuracy=exact_acc,
            near_accuracy=near_acc,
            by_assumption=by_assumption,
            label=data.get('label', ''),
            filename=data['name'],
        ))

    if not scaling_results:
        raise ValueError("No valid results computed")

    # Keep in original order (by label A, B, C, ...)

    # Compute pairwise comparison if multiple files
    pairwise_by_assumption: Optional[dict[str, PairwiseComparisonResult]] = None

    if len(scaling_results) > 1:
        # Validate that all EGF files reference the same EDF before pairwise comparison
        sources_match, source_warning = validate_egf_sources_match(egf_files)

        if not sources_match:
            print()
            print(source_warning)
            print()
            print("Skipping pairwise comparisons due to mismatched EDF sources.")
        else:
            print("\nComputing pairwise comparisons...")
            pairwise_by_assumption = {}

            # Build by_idx dict for pairwise analysis (use index to avoid collisions when n_value is same)
            # IMPORTANT: Align all files by target_id so the same index = same essay across files
            # This is required for the shared GT noise cache in bootstrap to work correctly

            # First, collect all data keyed by target_id for each file
            file_grades_by_target: list[dict[str, tuple[int, int]]] = []
            for data in file_data:
                grades_dict = {}
                for g in data['grades']:
                    if g['ground_truth'] is not None:
                        grades_dict[g['target_id']] = (g['predicted'], g['ground_truth'])
                file_grades_by_target.append(grades_dict)

            # Find common target_ids across all files
            common_targets = set(file_grades_by_target[0].keys())
            for grades_dict in file_grades_by_target[1:]:
                common_targets &= set(grades_dict.keys())
            common_targets = sorted(common_targets)

            print(f"  Using {len(common_targets)} essays common to all files for pairwise comparison")

            # Build aligned data for each file
            by_idx: dict[int, tuple[list[int], list[int], list[str]]] = {}
            stability_vectors_by_idx: dict[int, StabilityVector] = {}
            for idx, (data, grades_dict) in enumerate(zip(file_data, file_grades_by_target)):
                predicted = [grades_dict[tid][0] for tid in common_targets]
                ground_truth = [grades_dict[tid][1] for tid in common_targets]
                if predicted:
                    by_idx[idx] = (predicted, ground_truth, list(common_targets))
                    n_val = data['n_value']
                    stability_vectors_by_idx[idx] = stability_vectors.get(n_val, get_default_stability_vector())

            for assumption in ["optimistic", "expected", "pessimistic"]:
                teacher_noise = all_teacher_noise[assumption]

                win_matrix_qwk, avg_qwk = bootstrap_qwk_paired(
                    by_idx, stability_vectors_by_idx, teacher_noise,
                    n_iterations=2000, seed=seed,
                )

                win_matrix_essay = bootstrap_essay_level_paired(
                    by_idx, stability_vectors_by_idx, teacher_noise,
                    n_iterations=2000, seed=seed,
                )

                pairwise_by_assumption[assumption] = PairwiseComparisonResult(
                    n_values=list(range(len(file_data))),
                    win_matrix_qwk=win_matrix_qwk,
                    win_matrix_essay=win_matrix_essay,
                    n_iterations=2000,
                    avg_qwk_by_n=avg_qwk,
                    labels=labels,
                    legend=legend,
                )

                print(f"  {assumption}: computed pairwise matrix")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = egf_files[0].parent.name if egf_files else "analysis"
    experiment_output_dir = output_dir / f"{run_name}_{timestamp}"
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    # Build results object
    results = MultiEGFResults(
        egf_names=[f.name for f in egf_files],
        n_values=list(range(len(scaling_results))),  # Use indices for consistency
        per_file_results=scaling_results,
        pairwise_by_assumption=pairwise_by_assumption,
        source_edf_name=source_edf_name,
        grading_description=grading_description,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        labels=labels,
        legend=legend,
    )

    # Save report
    report_path = experiment_output_dir / "report.md"
    save_multi_egf_report(results, report_path)
    print(f"\nReport saved to: {report_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    save_multi_assumption_qwk_chart(scaling_results, experiment_output_dir)
    save_multi_assumption_grid_chart(scaling_results, experiment_output_dir)

    if pairwise_by_assumption:
        # Save heatmaps for all 3 assumptions
        for assumption in ["optimistic", "expected", "pessimistic"]:
            save_pairwise_heatmap(
                pairwise_by_assumption[assumption],
                experiment_output_dir,
                filename_suffix=assumption,
            )

    print(f"Output directory: {experiment_output_dir}")
    return results


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Resolve EGF files first
    egf_files: list[Path] = []

    if args.paths:
        # Explicit paths provided
        egf_files = resolve_egf_paths(args.paths, input_dir)

    elif args.auto:
        # Auto-detect latest run folder
        run_folder = find_latest_run_folder(input_dir)
        if run_folder:
            print(f"Auto-detected run folder: {run_folder}")
            egf_files = resolve_egf_paths([str(run_folder)], input_dir)
        else:
            print("Error: No run folders found.")
            print(f"  Looked in: {input_dir}")
            return

    else:
        print("Error: Provide EGF file(s)/folder(s) or use --auto")
        print(f"\nAvailable run folders in {input_dir}:")
        if input_dir.exists():
            for item in sorted(input_dir.iterdir()):
                if item.is_dir():
                    egf_count = len(list(item.glob("*.egf")))
                    if egf_count > 0:
                        print(f"  {item.name}/ ({egf_count} EGF files)")
        return

    if not egf_files:
        print("Error: No EGF files found")
        return

    print(f"Found {len(egf_files)} EGF file(s)")

    # Extract all teacher noise models from source EDF (via EGF reference)
    all_teacher_noise: dict[str, TeacherNoiseModel]
    source_edf_name: Optional[str] = None
    grading_description: Optional[str] = None

    if args.teacher_noise:
        # Single teacher noise override - use for all assumptions
        try:
            single_noise = parse_teacher_noise_json(args.teacher_noise)
            print(f"Using teacher noise override: {single_noise}")
            # Use same noise for all assumptions
            all_teacher_noise = {
                "optimistic": single_noise,
                "expected": single_noise,
                "pessimistic": single_noise,
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Invalid --teacher-noise JSON: {e}")
            return
    else:
        # Extract all 3 noise models from source EDF
        extracted, edf_path = extract_all_teacher_noise_from_egf(egf_files)
        if extracted:
            all_teacher_noise = extracted
            source_edf_name = edf_path.name if edf_path else None
        else:
            print("Warning: Could not extract teacher noise from EDF, using defaults")
            all_teacher_noise = get_default_all_teacher_noise()

    # Try to get grading description from EGF
    try:
        with EGF.open(egf_files[0]) as egf:
            grading_description = egf.grading_config.description
    except Exception:
        pass

    # Get default stability vectors (ideally from Exp1 data, but use defaults here)
    stability_vectors: dict[int, StabilityVector] = {}
    default_sv = get_default_stability_vector()
    for n in [5, 10, 15, 20, 25, 30]:
        stability_vectors[n] = default_sv.copy()

    # Run multi-assumption analysis
    analyse_egf_files_with_multi_assumption(
        egf_files=egf_files,
        output_dir=output_dir,
        all_teacher_noise=all_teacher_noise,
        stability_vectors=stability_vectors,
        seed=args.seed,
        source_edf_name=source_edf_name,
        grading_description=grading_description,
    )


if __name__ == "__main__":
    main()
