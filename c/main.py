#!/usr/bin/env python3
"""CLI tool for analyzing EGF grading results against EDF ground truth.

Usage:
    c <egf_files...> [edf_folder]        Analyze EGF file(s) (scans CWD for EDFs if no folder given)
    c -w -a <base_egf>                   Watch for new EGF files (scans CWD for EDFs)

Examples:
    c results.egf                        Analyze single EGF
    compare a.egf b.egf c.egf            Compare multiple EGFs
    c -w -a base.egf                     Watch mode
"""

import argparse
import sys
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from edf import EDF
from egf import EGF

from .core import (
    EGFData,
    Essay,
    load_egf_data,
    load_edf_teacher_noise,
    load_edf_ground_truth,
    build_essays,
    analyze_multiple_egf,
    get_default_teacher_noise,
    find_matching_edf,
    all_same_source,
    load_edf_submissions_detail,
    load_egf_grades_detail,
    load_egf_all_llm_calls,
    build_grades_table_data,
    ProcessedInput,
    MatchedEGF,
    load_egf_comparisons,
    build_comparison_tuples,
    ComparisonAccuracyResult,
)
from .edf_cache import EDFCache
from .bootstrap import (
    get_default_stability_vector,
    bootstrap_comparison_accuracy,
    bootstrap_comparison_accuracy_paired,
    compute_raw_comparison_accuracy,
)
from .html_output import save_html_report
from .core import FullAnalysisResult


def generate_summary_markdown(result: FullAnalysisResult) -> str:
    """Generate a markdown summary of analysis results."""
    lines = []
    lines.append("=" * 60)
    lines.append("Results Summary")
    lines.append("=" * 60)

    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        qwk = res.qwk_result
        lines.append("")
        lines.append(f"[{label}] {res.egf_name}")
        lines.append(f"    Raw QWK: {qwk.raw_qwk:.4f}")
        lines.append(f"    Exact Acc: {qwk.exact_accuracy:.1%} | Near Acc: {qwk.near_accuracy:.1%}")
        gt_mean, gt_lower, gt_upper = qwk.gt_noise_ci
        lines.append(f"    GT Noise CI: [{gt_lower:.3f}, {gt_upper:.3f}]")
        grading_mean, grading_lower, grading_upper = qwk.grading_noise_ci
        lines.append(f"    Grading Noise CI: [{grading_lower:.3f}, {grading_upper:.3f}]")
        sampling_mean, sampling_lower, sampling_upper = qwk.sampling_ci
        lines.append(f"    Sampling CI: [{sampling_lower:.3f}, {sampling_upper:.3f}]")
        combined_mean, combined_lower, combined_upper = qwk.combined_ci
        lines.append(f"    Combined CI: [{combined_lower:.3f}, {combined_upper:.3f}]")

    if result.comparison and len(result.labels) > 1:
        lines.append("")
        lines.append("Comparison Matrix P(Row > Col):")
        n = len(result.labels)
        header = "     " + "  ".join(f"{l:>5}" for l in result.labels[:n])
        lines.append(header)
        for i in range(n):
            row = f"{result.labels[i]:>4} "
            for j in range(n):
                if i == j:
                    row += "    - "
                else:
                    p = result.comparison.win_matrix.get((i, j), 0.5)
                    row += f" {p:4.0%} "
            lines.append(row)

    if result.comparison_accuracy:
        lines.append("")
        lines.append("Comparison Accuracy:")
        for egf_name, acc in result.comparison_accuracy.items():
            mean, lower, upper = acc.accuracy_ci
            lines.append(f"  {egf_name}: {mean:.1%} [{lower:.1%}, {upper:.1%}] (n={acc.n_comparisons})")

    lines.append("=" * 60)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze EGF grading results against EDF ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    c results.egf                        Analyze single EGF
    compare a.egf b.egf c.egf            Compare multiple EGFs
    c -w -a base.egf                     Watch mode (scans CWD for EDFs)
        """,
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help="EGF file(s) and optional EDF folder (defaults to CWD)",
    )

    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Watch mode: monitor for new EGF files",
    )

    parser.add_argument(
        "-a", "--against",
        type=str,
        help="Base EGF file for watch mode comparison",
    )

    parser.add_argument(
        "--noise",
        type=str,
        choices=["optimistic", "expected", "pessimistic"],
        default="expected",
        help="Noise assumption for teacher variance (default: expected)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output HTML file path (default: auto-generated)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Don't open HTML report in browser",
    )

    parser.add_argument(
        "--edf-path",
        type=str,
        default=None,
        help="Directory to search for EDF files (default: CWD)",
    )

    return parser.parse_args()


def resolve_paths(paths: list[str]) -> tuple[list[Path], Optional[Path]]:
    """Resolve input paths into EGF files and EDF directory.

    Returns:
        Tuple of (egf_files, edf_directory)
    """
    if not paths:
        return [], None

    resolved_paths = [Path(p) for p in paths]

    directories = [p for p in resolved_paths if p.is_dir()]
    egf_files = [p for p in resolved_paths if p.is_file() and p.suffix == ".egf"]

    expanded_egf_files = list(egf_files)
    for p in resolved_paths:
        if p.is_dir():
            expanded_egf_files.extend(p.glob("*.egf"))

    if not directories:
        edf_directory = Path.cwd()
    else:
        edf_directory = directories[-1]

    expanded_egf_files = list(set(expanded_egf_files))

    return expanded_egf_files, edf_directory


def find_egf_files_in_args(paths: list[str]) -> tuple[list[Path], Path]:
    """Parse arguments to separate EGF files from EDF folder.

    If a directory is provided, it's used as the EDF folder.
    If no directory is provided, CWD is used as the EDF folder.
    All .egf files are collected for analysis.
    """
    if not paths:
        print("Error: No paths provided", file=sys.stderr)
        print("Usage: c <egf_files...> [edf_folder]", file=sys.stderr)
        sys.exit(1)

    resolved = [Path(p) for p in paths]

    directories = [p for p in resolved if p.is_dir()]

    # Default to CWD if no directory provided
    if not directories:
        edf_folder = Path.cwd()
    else:
        edf_folder = directories[-1]

    egf_files = []
    for p in resolved:
        if p.is_dir():
            # If it's the EDF folder, don't scan it for EGF files
            if p != edf_folder:
                egf_files.extend(sorted(p.glob("*.egf")))
        elif p.is_file() and p.suffix == ".egf":
            egf_files.append(p)

    egf_files = list(dict.fromkeys(egf_files))

    return egf_files, edf_folder


def check_missing_edfs(egf_data_list: list[EGFData], edf_cache: EDFCache) -> list[EGFData]:
    """Check for EGF files with missing EDF matches and warn."""
    valid = []
    for egf_data in egf_data_list:
        edf_path = find_matching_edf(egf_data, edf_cache)
        if edf_path:
            valid.append(egf_data)
        else:
            print(f"Warning: No matching EDF for {egf_data.name} (hash: {egf_data.source_hash[:20]}...)")
            print(f"  Dropping from analysis")

    return valid


def process_input_queue(cli_args: list[str]) -> ProcessedInput:
    """Process CLI args using queue-based discovery.

    Recursively processes folders, detects ephemeral EGFs (unzipped EGF folders),
    and collects all .egf files.
    """
    queue = deque(cli_args)
    egf_files: list[Path] = []
    ephemeral_egfs: list[Path] = []
    warnings: list[str] = []

    while queue:
        item = queue.popleft()
        path = Path(item)

        if not path.exists():
            warnings.append(f"Skipping non-existent: {item}")
            continue

        if path.is_file():
            if path.suffix == ".egf":
                egf_files.append(path)
            else:
                warnings.append(f"Skipping non-EGF file: {item}")
            continue

        if path.is_dir():
            # Try EGF.open() to detect ephemeral EGF (unzipped EGF folder)
            try:
                with EGF.open(path) as _egf:
                    ephemeral_egfs.append(path)
                    warnings.append(
                        f"'{path}' parsed as ephemeral EGF. "
                        f"Delete manifest.json to treat as folder."
                    )
                    continue
            except Exception:
                pass  # Not an EGF, treat as folder

            # Expand folder contents into queue
            for child in sorted(path.iterdir()):
                if child.is_dir() or child.suffix == ".egf":
                    queue.append(str(child))

    return ProcessedInput(
        egf_paths=list(dict.fromkeys(egf_files)),  # Remove duplicates, preserve order
        ephemeral_egf_paths=list(dict.fromkeys(ephemeral_egfs)),
        warnings=warnings,
    )


def scan_edf_files_direct(search_root: Path) -> dict[str, Path]:
    """Recursively scan for EDF files, returning hash->path index.

    Does NOT use cache - computes hash for each EDF directly.
    Also detects unpacked EDF directories (with manifest.json + submissions/).
    """
    hash_to_path: dict[str, Path] = {}

    # Find all .edf files
    for edf_path in search_root.rglob("*.edf"):
        try:
            with EDF.open(edf_path) as edf:
                if edf.content_hash:
                    hash_to_path[edf.content_hash] = edf_path
        except Exception:
            continue

    # Also check unpacked EDF directories (manifest.json + submissions/)
    for manifest in search_root.rglob("manifest.json"):
        parent = manifest.parent
        if (parent / "submissions").is_dir():
            try:
                with EDF.open(parent) as edf:
                    if edf.content_hash:
                        hash_to_path[edf.content_hash] = parent
            except Exception:
                continue

    return hash_to_path


def match_egfs_to_edfs(
    egf_paths: list[Path],
    edf_index: dict[str, Path],
) -> tuple[list[MatchedEGF], list[str]]:
    """Load EGFs and match each to its EDF by source hash.

    Returns:
        Tuple of (matched EGFs, error messages for unmatched ones)
    """
    matched: list[MatchedEGF] = []
    errors: list[str] = []

    for egf_path in egf_paths:
        try:
            egf_data = load_egf_data(egf_path)
        except Exception as e:
            errors.append(f"Failed to load {egf_path.name}: {e}")
            continue

        edf_path = edf_index.get(egf_data.source_hash)
        if edf_path is None:
            errors.append(
                f"No matching EDF for {egf_data.name} "
                f"(hash: {egf_data.source_hash[:16]}...)"
            )
            continue

        matched.append(MatchedEGF(egf_data=egf_data, edf_path=edf_path))

    return matched, errors


def run_stats_mode(
    matched_egfs: list[MatchedEGF],
    noise_assumption: str,
    output_path: Optional[Path],
    seed: int,
    quiet: bool,
) -> list[Path]:
    """Generate separate HTML report for each EGF (stats only mode)."""
    output_paths: list[Path] = []

    for i, matched in enumerate(matched_egfs, 1):
        print(f"\n[{i}/{len(matched_egfs)}] Analyzing {matched.egf_data.name}...")
        print(f"  EDF: {matched.edf_path.name}")

        # Run single-file analysis
        run_analysis(
            [matched.egf_data.path],
            matched.edf_path.parent,
            noise_assumption,
            output_path,  # Will auto-generate filename if None
            seed,
            quiet,
        )

    return output_paths


def run_analysis(
    egf_files: list[Path],
    edf_folder: Path,
    noise_assumption: str,
    output_path: Optional[Path],
    seed: int,
    quiet: bool = False,
) -> None:
    """Run the main analysis."""
    print(f"Loading {len(egf_files)} EGF file(s)...")
    egf_data_list = []
    for path in egf_files:
        try:
            data = load_egf_data(path)
            print(f"  {path.name}: {len(data.grades)} grades")
            egf_data_list.append(data)
        except Exception as e:
            print(f"  Warning: Failed to load {path.name}: {e}", file=sys.stderr)

    if not egf_data_list:
        print("Error: No valid EGF files loaded", file=sys.stderr)
        sys.exit(1)

    print(f"\nScanning EDF directory: {edf_folder}")
    edf_cache = EDFCache(edf_folder)

    egf_data_list = check_missing_edfs(egf_data_list, edf_cache)
    if not egf_data_list:
        print("Error: No EGF files with matching EDFs", file=sys.stderr)
        sys.exit(1)

    first_egf = egf_data_list[0]
    edf_path = find_matching_edf(first_egf, edf_cache)

    if not edf_path:
        print("Error: No matching EDF found. EDF is required for ground truth.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading teacher noise from: {edf_path.name}")
    edf_data = load_edf_teacher_noise(edf_path)
    teacher_noise_all = edf_data.teacher_noise
    edf_name = edf_data.name

    print(f"Loading ground truth from: {edf_path.name}")
    ground_truth = load_edf_ground_truth(edf_path)

    # Build essays by combining predicted grades (EGF) with ground truth (EDF)
    essays_list = [build_essays(egf, ground_truth) for egf in egf_data_list]

    teacher_noise = teacher_noise_all.get(noise_assumption, teacher_noise_all["expected"])

    stability_vectors = {i: get_default_stability_vector() for i in range(len(egf_data_list))}

    print(f"\nAnalyzing with noise assumption: {noise_assumption}")

    result = analyze_multiple_egf(
        egf_data_list,
        essays_list,
        teacher_noise,
        stability_vectors,
        edf_name,
        noise_assumption,
        seed,
    )

    if len(egf_data_list) > 1:
        if all_same_source(egf_data_list):
            print(f"  All files reference same source EDF - computing comparison matrix")
        else:
            print(f"  Warning: Files reference different EDFs - skipping comparison matrix")

    # Build grades table data for the interactive HTML table
    if edf_path:
        try:
            print(f"\nLoading submission details for grades table...")
            edf_submissions = load_edf_submissions_detail(edf_path, noise_assumption)

            egf_grades_list = []
            egf_all_llm_calls_list = []
            for egf_data in egf_data_list:
                egf_grades = load_egf_grades_detail(egf_data.path)
                egf_grades_list.append((egf_data.name, egf_grades))
                # Load all LLM calls (including comparison calls)
                all_calls = load_egf_all_llm_calls(egf_data.path)
                egf_all_llm_calls_list.append((egf_data.name, all_calls))

            max_grade = egf_data_list[0].max_grade if egf_data_list else 40
            result.grades_table = build_grades_table_data(
                edf_submissions,
                egf_grades_list,
                result.labels,
                max_grade,
                noise_assumption,
                egf_all_llm_calls_list=egf_all_llm_calls_list,
            )
            print(f"  Loaded {len(edf_submissions)} submissions")
        except Exception as e:
            print(f"  Warning: Failed to build grades table: {e}", file=sys.stderr)

    # Load comparisons for each EGF and compute accuracy
    if edf_path and result.grades_table:
        print(f"\nLoading comparisons from EGF files...")
        edf_submission_ids = set(edf_submissions.keys())
        egf_comparisons: dict[str, dict[str, list]] = {}
        comparison_accuracy: dict[str, ComparisonAccuracyResult] = {}
        all_have_comparisons = True
        comparison_tuples_by_idx: dict[int, list[tuple[int, int, str]]] = {}

        for idx, egf_data in enumerate(egf_data_list):
            try:
                comparisons = load_egf_comparisons(egf_data.path, edf_submission_ids)
                egf_comparisons[egf_data.name] = comparisons

                # Build comparison tuples for accuracy calculation
                # Flatten all comparisons for this EGF
                all_comparisons = []
                for sub_id, comp_list in comparisons.items():
                    all_comparisons.extend(comp_list)

                # Filter to only EDF-to-EDF comparisons (not external)
                edf_comparisons = [c for c in all_comparisons if not c.is_external]
                unique_comparisons = {c.comparison_id: c for c in edf_comparisons}
                edf_comparisons = list(unique_comparisons.values())

                if edf_comparisons:
                    # Build tuples for bootstrap
                    tuples = build_comparison_tuples(edf_comparisons, ground_truth)
                    if tuples:
                        comparison_tuples_by_idx[idx] = tuples
                        # Compute accuracy for this EGF
                        acc_ci = bootstrap_comparison_accuracy(
                            tuples,
                            teacher_noise,
                            n_iterations=1000,
                            seed=seed,
                            min_grade=0,
                            max_grade=max_grade,
                        )
                        n_external = len(all_comparisons) - len(edf_comparisons)
                        raw_acc, _, _ = compute_raw_comparison_accuracy(tuples)
                        comparison_accuracy[egf_data.name] = ComparisonAccuracyResult(
                            raw_accuracy=raw_acc,
                            n_comparisons=len(tuples),
                            n_excluded_external=n_external,
                            accuracy_ci=acc_ci,
                        )
                        print(f"  {egf_data.name}: {len(tuples)} comparisons, raw accuracy {raw_acc:.1%}, CI [{acc_ci[1]:.1%}, {acc_ci[2]:.1%}]")
                    else:
                        all_have_comparisons = False
                        print(f"  {egf_data.name}: No valid comparison tuples")
                else:
                    all_have_comparisons = False
                    print(f"  {egf_data.name}: No EDF-to-EDF comparisons")
            except Exception as e:
                all_have_comparisons = False
                print(f"  Warning: Failed to load comparisons for {egf_data.name}: {e}", file=sys.stderr)

        # Store comparisons in grades table
        if egf_comparisons:
            result.grades_table.egf_comparisons = egf_comparisons

        # Store comparison accuracy results
        if comparison_accuracy:
            result.comparison_accuracy = comparison_accuracy

        # Compute NxN paired comparison accuracy if all files have comparisons
        if all_have_comparisons and len(comparison_tuples_by_idx) > 1:
            print(f"\nComputing paired comparison accuracy matrix...")
            try:
                matrix, per_file = bootstrap_comparison_accuracy_paired(
                    comparison_tuples_by_idx,
                    teacher_noise,
                    n_iterations=2000,
                    seed=seed,
                    min_grade=0,
                    max_grade=max_grade,
                )
                result.comparison_accuracy_matrix = matrix
                print(f"  Computed {len(matrix)} pairwise comparisons")
            except Exception as e:
                print(f"  Warning: Failed to compute comparison matrix: {e}", file=sys.stderr)

    # Generate summary and store it in the result for HTML embedding
    result.summary_markdown = generate_summary_markdown(result)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(egf_data_list) == 1:
            output_path = Path(f"{egf_data_list[0].name}_{timestamp}.html")
        else:
            output_path = Path(f"comparison_{timestamp}.html")

    print(f"\nGenerating HTML report...")
    save_html_report(result, output_path, noise_assumption)
    print(f"Output: {output_path}")

    # Open in browser
    if not quiet:
        webbrowser.open(output_path.resolve().as_uri())

    # Print summary to terminal
    print("\n" + result.summary_markdown)


def main() -> None:
    args = parse_args()

    if args.watch:
        if not args.against:
            print("Error: -w/--watch requires -a/--against=<egf_file>", file=sys.stderr)
            sys.exit(1)

        base_path = Path(args.against)
        if not base_path.exists():
            print(f"Error: Base file not found: {args.against}", file=sys.stderr)
            sys.exit(1)

        # Default to CWD if no directory provided
        if args.paths:
            directories = [Path(p) for p in args.paths if Path(p).is_dir()]
            edf_folder = directories[-1] if directories else Path.cwd()
        else:
            edf_folder = Path.cwd()

        from .watch import run_watch_mode
        run_watch_mode(base_path, edf_folder, args.noise, args.quiet)

    else:
        # Step 1: Process input queue
        print("Processing input arguments...")
        processed = process_input_queue(args.paths)

        for warning in processed.warnings:
            print(f"  Warning: {warning}", file=sys.stderr)

        all_egf_paths = processed.egf_paths + processed.ephemeral_egf_paths

        if not all_egf_paths:
            print("Error: No EGF files found", file=sys.stderr)
            print("Usage: c <egf_files...> [--edf-path <folder>]", file=sys.stderr)
            sys.exit(1)

        print(f"  Found {len(all_egf_paths)} EGF file(s)")

        # Step 2: Scan for EDFs (no cache)
        edf_search_root = Path(args.edf_path) if args.edf_path else Path.cwd()
        print(f"\nScanning for EDF files in: {edf_search_root}")
        edf_index = scan_edf_files_direct(edf_search_root)
        print(f"  Found {len(edf_index)} EDF file(s)")

        # Step 3: Match EGFs to EDFs
        print("\nMatching EGF files to EDFs...")
        matched_egfs, match_errors = match_egfs_to_edfs(all_egf_paths, edf_index)

        for error in match_errors:
            print(f"  Error: {error}", file=sys.stderr)

        if not matched_egfs:
            print("Error: No EGF files with matching EDFs", file=sys.stderr)
            sys.exit(1)

        print(f"  Matched {len(matched_egfs)} EGF file(s)")

        # Step 4: Determine mode and run analysis
        unique_edfs = set(m.edf_path for m in matched_egfs)
        output_path = Path(args.output) if args.output else None

        if len(matched_egfs) > 1 and len(unique_edfs) == 1:
            # Stats + NxN comparison mode
            print(f"\nMode: Stats + NxN Comparison (all EGFs share 1 EDF)")
            egf_files = [m.egf_data.path for m in matched_egfs]
            edf_folder = matched_egfs[0].edf_path.parent
            run_analysis(egf_files, edf_folder, args.noise, output_path, args.seed, args.quiet)
        else:
            # Stats only mode
            if len(unique_edfs) > 1:
                print(f"\nWarning: {len(unique_edfs)} different EDFs detected", file=sys.stderr)
                print("  NxN comparison requires same dataset. Generating separate reports.", file=sys.stderr)
            print(f"\nMode: Stats Only (generating {len(matched_egfs)} report(s))")
            run_stats_mode(matched_egfs, args.noise, output_path, args.seed, args.quiet)


if __name__ == "__main__":
    main()
