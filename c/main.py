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
from datetime import datetime
from pathlib import Path
from typing import Optional

from .core import (
    EGFData,
    load_egf_data,
    load_edf_teacher_noise,
    analyze_multiple_egf,
    get_default_teacher_noise,
    find_matching_edf,
    all_same_source,
)
from .edf_cache import EDFCache
from .bootstrap import get_default_stability_vector
from .html_output import save_html_report


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


def run_analysis(
    egf_files: list[Path],
    edf_folder: Path,
    noise_assumption: str,
    output_path: Optional[Path],
    seed: int,
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

    teacher_noise_all = get_default_teacher_noise()
    edf_name = None
    if edf_path:
        print(f"\nLoading teacher noise from: {edf_path.name}")
        try:
            edf_data = load_edf_teacher_noise(edf_path)
            teacher_noise_all = edf_data.teacher_noise
            edf_name = edf_data.name
        except Exception as e:
            print(f"  Warning: Failed to load EDF, using defaults: {e}", file=sys.stderr)

    teacher_noise = teacher_noise_all.get(noise_assumption, teacher_noise_all["expected"])

    stability_vectors = {i: get_default_stability_vector() for i in range(len(egf_data_list))}

    print(f"\nAnalyzing with noise assumption: {noise_assumption}")

    result = analyze_multiple_egf(
        egf_data_list,
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

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(egf_data_list) == 1:
            output_path = Path(f"{egf_data_list[0].name}_{timestamp}.html")
        else:
            output_path = Path(f"comparison_{timestamp}.html")

    print(f"\nGenerating HTML report...")
    save_html_report(result, output_path, noise_assumption)
    print(f"Output: {output_path}")

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        qwk = res.qwk_result
        print(f"\n[{label}] {res.egf_name}")
        print(f"    Raw QWK: {qwk.raw_qwk:.4f}")
        print(f"    Exact Acc: {qwk.exact_accuracy:.1%} | Near Acc: {qwk.near_accuracy:.1%}")
        gt_mean, gt_lower, gt_upper = qwk.gt_noise_ci
        print(f"    GT Noise CI: [{gt_lower:.3f}, {gt_upper:.3f}]")
        grading_mean, grading_lower, grading_upper = qwk.grading_noise_ci
        print(f"    Grading Noise CI: [{grading_lower:.3f}, {grading_upper:.3f}]")
        sampling_mean, sampling_lower, sampling_upper = qwk.sampling_ci
        print(f"    Sampling CI: [{sampling_lower:.3f}, {sampling_upper:.3f}]")

    if result.comparison and len(result.labels) > 1:
        print("\nComparison Matrix P(Row > Col):")
        n = len(result.labels)
        header = "     " + "  ".join(f"{l:>5}" for l in result.labels[:n])
        print(header)
        for i in range(n):
            row = f"{result.labels[i]:>4} "
            for j in range(n):
                if i == j:
                    row += "    - "
                else:
                    p = result.comparison.win_matrix.get((i, j), 0.5)
                    row += f" {p:4.0%} "
            print(row)

    print("="*60)


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
        run_watch_mode(base_path, edf_folder, args.noise)

    else:
        egf_files, edf_folder = find_egf_files_in_args(args.paths)

        if not egf_files:
            print("Error: No EGF files found", file=sys.stderr)
            print("Usage: c <egf_files...> <edf_folder>", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output) if args.output else None

        run_analysis(egf_files, edf_folder, args.noise, output_path, args.seed)


if __name__ == "__main__":
    main()
