"""Report generation for experiment analysis."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class StabilityResult:
    """Results from Experiment 1 for a single N value."""
    n_anchors: int
    stability_vector: dict[int, float]
    standard_errors: dict[int, float]
    n_samples: int


@dataclass
class ScalingResult:
    """Results from Experiment 2 for a single N value."""
    n_anchors: int
    raw_qwk: float
    mean_qwk: float
    # Combined noise CI (all sources: sampling + LLM + teacher)
    lower_ci: float
    upper_ci: float
    # Sampling noise only CI (no noise models applied)
    ci_sampling_only_lower: float
    ci_sampling_only_upper: float
    # LLM noise only CI (no resampling)
    ci_llm_only_lower: float
    ci_llm_only_upper: float
    # Teacher noise only CI (no resampling)
    ci_teacher_only_lower: float
    ci_teacher_only_upper: float
    exact_accuracy: float
    near_accuracy: float
    n_targets: int


@dataclass
class ScalingResultByAssumption:
    """QWK results for a single file under all 3 noise assumptions."""
    n_anchors: int
    n_targets: int
    raw_qwk: float
    exact_accuracy: float
    near_accuracy: float
    # Results by assumption (optimistic, expected, pessimistic)
    # Each contains: mean_qwk, lower_ci, upper_ci for each CI type
    by_assumption: dict[str, dict[str, tuple[float, float, float]]]
    # Keys: "optimistic", "expected", "pessimistic"
    # Values: dict with keys:
    #   "sampling_only": (mean, lower, upper)
    #   "llm_only": (mean, lower, upper)
    #   "teacher_only": (mean, lower, upper)
    #   "combined": (mean, lower, upper)
    # Short label for display (A, B, C, etc.)
    label: str = ""
    # Original filename for legend
    filename: str = ""


@dataclass
class MultiEGFResults:
    """Results from analyzing multiple EGF files."""
    egf_names: list[str]
    n_values: list[int]
    # Per-file results keyed by assumption
    per_file_results: list[ScalingResultByAssumption]
    # Pairwise comparison matrices (by assumption)
    pairwise_by_assumption: Optional[dict[str, "PairwiseComparisonResult"]]
    # Source EDF info
    source_edf_name: Optional[str]
    grading_description: Optional[str]
    timestamp: str
    # Labels for each file (A, B, C, etc.)
    labels: list[str] = None  # type: ignore
    # Legend mapping labels to filenames
    legend: dict[str, str] = None  # type: ignore

    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.legend is None:
            self.legend = {}


@dataclass
class PairwiseComparisonResult:
    """Results from paired bootstrap comparison across files/N values.

    Computes P(X beats Y) by applying the same scenario to all files.
    """
    n_values: list[int]
    win_matrix_qwk: dict[tuple[int, int], float]  # P(n_i > n_j) based on QWK
    win_matrix_essay: dict[tuple[int, int], float]  # P(n_i > n_j) based on essay-level
    n_iterations: int
    avg_qwk_by_n: dict[int, float]
    # Optional labels for display (if not using n_values)
    labels: list[str] = None  # type: ignore
    # Legend for display
    legend: dict[str, str] = None  # type: ignore

    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.legend is None:
            self.legend = {}


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    stability_results: list[StabilityResult]  # Experiment 1: permutation stability
    scaling_results: list[ScalingResult]
    pairwise_results: Optional[PairwiseComparisonResult]
    dataset_name: str
    model_name: str
    master_seed: int
    timestamp: str
    throughput_stats: dict[str, Any]
    # Source of noise models for documentation
    llm_noise_source: str = "Experiment 1 data"
    teacher_noise_source: str = "Source EDF file"
    # Experiment 3: content stability (different anchor sets)
    content_stability_results: list[StabilityResult] = None  # type: ignore

    def __post_init__(self):
        if self.content_stability_results is None:
            self.content_stability_results = []


def _format_stability_table(results: list[StabilityResult]) -> str:
    """Format Experiment 1 stability results as a Markdown table."""
    if not results:
        return "_No stability results available._\n"
    all_deviations: set[int] = set()
    for r in results:
        all_deviations.update(r.stability_vector.keys())
    deviations = sorted(all_deviations)
    header = "| N | " + " | ".join(f"Dev {d:+d}" for d in deviations) + " |"
    separator = "|---| " + " | ".join("---" for _ in deviations) + " |"
    rows = []
    for r in results:
        row_parts = [f"| {r.n_anchors}"]
        for d in deviations:
            prob = r.stability_vector.get(d, 0.0)
            se = r.standard_errors.get(d, 0.0)
            row_parts.append(f"{prob:.3f} ± {se:.3f}")
        rows.append(" | ".join(row_parts) + " |")
    return "\n".join([header, separator] + rows) + "\n"


def _format_scaling_table(results: list[ScalingResult]) -> str:
    """Format Experiment 2 scaling results as a Markdown table."""
    if not results:
        return "_No scaling results available._\n"
    header = "| N | Raw QWK | Mean QWK | CI (Sampling) | CI (LLM) | CI (Teacher) | CI (All) | Exact Acc | Near Acc |"
    separator = "|---|---------|----------|---------------|----------|--------------|----------|-----------|----------|"
    rows = []
    for r in results:
        ci_sampling = f"[{r.ci_sampling_only_lower:.3f}, {r.ci_sampling_only_upper:.3f}]"
        ci_llm = f"[{r.ci_llm_only_lower:.3f}, {r.ci_llm_only_upper:.3f}]"
        ci_teacher = f"[{r.ci_teacher_only_lower:.3f}, {r.ci_teacher_only_upper:.3f}]"
        ci_all = f"[{r.lower_ci:.3f}, {r.upper_ci:.3f}]"
        row = (f"| {r.n_anchors} | {r.raw_qwk:.3f} | {r.mean_qwk:.3f} | {ci_sampling} | {ci_llm} | "
               f"{ci_teacher} | {ci_all} | {r.exact_accuracy:.1%} | {r.near_accuracy:.1%} |")
        rows.append(row)
    return "\n".join([header, separator] + rows) + "\n"


def _format_pairwise_matrix(
    n_values: list[int],
    win_matrix: dict[tuple[int, int], float],
    title: str,
    labels: Optional[list[str]] = None,
) -> list[str]:
    """Helper to format a single pairwise matrix.

    Args:
        n_values: List of N values (used as keys in win_matrix)
        win_matrix: Dict mapping (n_i, n_j) -> P(i beats j)
        title: Title for the matrix
        labels: Optional short labels to use instead of N values for display
    """
    lines = [f"### {title}\n"]

    # Use labels if provided, otherwise use N values
    display_labels = labels if labels else [f"N={n}" for n in n_values]

    # Build header
    header = "|   |" + "".join(f" {lbl} |" for lbl in display_labels)
    separator = "|---|" + "".join("---|" for _ in display_labels)

    lines.append(header)
    lines.append(separator)

    # Build rows
    for idx_i, n_i in enumerate(n_values):
        lbl_i = display_labels[idx_i]
        row_parts = [f"| **{lbl_i}**"]
        for idx_j, n_j in enumerate(n_values):
            if n_i == n_j:
                row_parts.append(" - ")
            else:
                p_win = win_matrix.get((n_i, n_j), 0)
                # Highlight if significantly better (>60%) or worse (<40%)
                if p_win > 0.60:
                    row_parts.append(f" **{p_win:.0%}** ")
                elif p_win < 0.40:
                    row_parts.append(f" _{p_win:.0%}_ ")
                else:
                    row_parts.append(f" {p_win:.0%} ")
        lines.append("|".join(row_parts) + "|")

    return lines


def _format_pairwise_table(result: Optional[PairwiseComparisonResult]) -> str:
    """Format pairwise comparison results as Markdown tables.

    Creates two matrices: one for QWK-based wins, one for essay-level wins.
    """
    if result is None:
        return "_No pairwise comparison results available._\n"

    n_values = result.n_values
    labels = result.labels if result.labels else None
    lines = []

    # QWK-based comparison
    lines.extend(_format_pairwise_matrix(
        n_values, result.win_matrix_qwk,
        "QWK-Based: P(Row has higher QWK than Column)",
        labels=labels,
    ))
    lines.append("")

    # Essay-level comparison
    lines.extend(_format_pairwise_matrix(
        n_values, result.win_matrix_essay,
        "Essay-Level: P(Row is equal or closer to GT per essay)",
        labels=labels,
    ))
    lines.append("")

    # Add average QWK summary with labels
    lines.append("**Average QWK (across all scenarios):**")
    for idx, n in enumerate(n_values):
        lbl = labels[idx] if labels else f"N={n}"
        lines.append(f"- {lbl}: {result.avg_qwk_by_n[n]:.3f}")

    lines.append("")
    lines.append(f"_Based on {result.n_iterations:,} paired bootstrap iterations._")
    lines.append("_Values > 50% indicate the row is more likely better than the column._")
    lines.append("_**Bold** = strong evidence (>60%), *italic* = weak evidence (<40%)._")

    return "\n".join(lines) + "\n"


def _format_throughput_section(stats: dict[str, Any]) -> str:
    """Format throughput statistics section."""
    if not stats:
        return "_No throughput statistics available._\n"
    lines = [
        f"- **Duration**: {stats.get('duration_seconds', 0):.1f}s",
        f"- **Total calls**: {stats.get('total_calls', 0)} "
        f"({stats.get('api_calls', 0)} API, {stats.get('cached_calls', 0)} cached)",
        f"- **Avg tokens/sec**: {stats.get('avg_tokens_per_second', 0):.0f}",
        f"- **Avg chars/sec**: {stats.get('avg_chars_per_second', 0):.0f}",
        f"- **Avg latency**: {stats.get('avg_latency_ms', 0):.0f}ms",
        f"- **Peak concurrent**: {stats.get('peak_concurrent_requests', 0)}",
    ]
    return "\n".join(lines) + "\n"


def _format_multi_assumption_table(result: ScalingResultByAssumption) -> str:
    """Format QWK results table showing all 3 noise assumptions and 4 CI types.

    Creates a 12-row table (4 CI types x 3 assumptions) for a single file/N value.
    """
    # Use label if available, otherwise use N value
    title = result.label if result.label else f"N = {result.n_anchors}"
    lines = [
        f"### {title} ({result.n_targets} essays)\n",
        f"**Raw QWK:** {result.raw_qwk:.4f} | "
        f"**Exact Acc:** {result.exact_accuracy:.1%} | "
        f"**Near Acc:** {result.near_accuracy:.1%}\n",
        "",
        "| CI Type | Assumption | Mean QWK | 95% CI |",
        "|---------|------------|----------|--------|",
    ]

    ci_type_names = {
        "sampling_only": "Sampling Only",
        "llm_only": "LLM Noise",
        "teacher_only": "Teacher Noise",
        "combined": "All Combined",
    }

    for ci_type, ci_label in ci_type_names.items():
        for assumption in ["optimistic", "expected", "pessimistic"]:
            if assumption in result.by_assumption:
                ci_data = result.by_assumption[assumption].get(ci_type)
                if ci_data:
                    mean_qwk, lower, upper = ci_data
                    ci_str = f"[{lower:.3f}, {upper:.3f}]"
                    lines.append(
                        f"| {ci_label} | {assumption.capitalize()} | {mean_qwk:.4f} | {ci_str} |"
                    )

    return "\n".join(lines) + "\n"


def _format_multi_egf_summary_table(results: list[ScalingResultByAssumption]) -> str:
    """Format summary table for multiple EGF files."""
    lines = [
        "## Summary\n",
        "| Label | Raw QWK | Exact Acc | Near Acc | CI (Expected, Combined) |",
        "|-------|---------|-----------|----------|-------------------------|",
    ]

    for r in results:
        # Get expected/combined CI as representative
        expected = r.by_assumption.get("expected", {})
        combined = expected.get("combined", (r.raw_qwk, r.raw_qwk, r.raw_qwk))
        ci_str = f"[{combined[1]:.3f}, {combined[2]:.3f}]"
        # Use label if available, otherwise N value
        display = r.label if r.label else f"N={r.n_anchors}"
        lines.append(
            f"| **{display}** | {r.raw_qwk:.4f} | {r.exact_accuracy:.1%} | "
            f"{r.near_accuracy:.1%} | {ci_str} |"
        )

    return "\n".join(lines) + "\n"


def generate_multi_egf_report(results: MultiEGFResults) -> str:
    """Generate a complete Markdown report for multi-EGF analysis."""
    sections = [
        "# EGF Analysis Report\n",
        f"**Generated:** {results.timestamp}\n",
    ]

    if results.source_edf_name:
        sections.append(f"**Source EDF:** {results.source_edf_name}\n")
    if results.grading_description:
        sections.append(f"**Grading:** {results.grading_description}\n")

    sections.append(f"**Files analyzed:** {len(results.egf_names)}\n")

    # Add legend if we have labels
    if results.legend:
        sections.append("\n## Legend\n")
        for label, filename in results.legend.items():
            sections.append(f"- **{label}**: `{filename}`\n")
    else:
        # Fallback to N values display
        sections.append(f"**N values:** {', '.join(str(n) for n in results.n_values)}\n")

    sections.append("\n---\n")

    # Summary table
    sections.append(_format_multi_egf_summary_table(results.per_file_results))

    # Visualizations right after summary - show all 3 assumptions
    sections.append("\n## Visualizations\n")
    sections.append("### QWK by Noise Assumption\n")
    sections.append("![QWK by Assumption](qwk_by_assumption.png)\n")

    if results.pairwise_by_assumption:
        sections.append("\n### Pairwise Comparison Heatmaps\n")
        sections.append("#### Optimistic Assumption\n")
        sections.append("![Pairwise Heatmap Optimistic](pairwise_heatmap_optimistic.png)\n")
        sections.append("#### Expected Assumption\n")
        sections.append("![Pairwise Heatmap Expected](pairwise_heatmap_expected.png)\n")
        sections.append("#### Pessimistic Assumption\n")
        sections.append("![Pairwise Heatmap Pessimistic](pairwise_heatmap_pessimistic.png)\n")

    # Detailed per-file results
    sections.append("\n---\n")
    sections.append("## Detailed Results\n")
    sections.append(
        "Shows QWK confidence intervals under 3 noise assumptions "
        "(optimistic/expected/pessimistic) and 4 CI types.\n\n"
    )

    for r in results.per_file_results:
        sections.append(_format_multi_assumption_table(r))
        sections.append("\n")

    # Pairwise comparison if available
    if results.pairwise_by_assumption:
        sections.append("\n---\n")
        sections.append("## Pairwise Comparison\n")
        sections.append(
            "P(Row N > Column N) under different noise assumptions.\n\n"
        )

        for assumption, pairwise in results.pairwise_by_assumption.items():
            sections.append(f"### {assumption.capitalize()} Assumption\n")
            sections.append(_format_pairwise_table(pairwise))
            sections.append("\n")

    return "\n".join(sections)


def save_multi_egf_report(results: MultiEGFResults, output_path: Path) -> None:
    """Save multi-EGF analysis report to file."""
    report = generate_multi_egf_report(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def generate_markdown_report(results: AnalysisResults) -> str:
    """Generate a complete Markdown report from analysis results."""
    sections = [
        "# Diparative Grading Benchmark Report\n",
        f"**Generated:** {results.timestamp}\n",
        f"**Dataset:** {results.dataset_name}\n",
        f"**Model:** {results.model_name}\n",
        f"**Master Seed:** {results.master_seed}\n",
        "\n---\n",
        "## Throughput Statistics\n",
        _format_throughput_section(results.throughput_stats),
        "\n---\n",
        "## Experiment 1: Permutation Stability\n",
        "Measures how anchor essay ordering affects predicted grades.\n",
        "Values show probability ± SE of each deviation from the mode grade.\n\n",
        _format_stability_table(results.stability_results),
    ]

    # Add stability histogram images
    if results.stability_results:
        sections.append("\n### Deviation Histograms\n")
        for sr in results.stability_results:
            sections.append(f"![Stability Histogram N={sr.n_anchors}](stability_histogram_n{sr.n_anchors}.png)\n")

    sections.extend([
        "\n---\n",
        "## Experiment 2: Scaling N (Grading Quality)\n",
        "Tests grading quality as number of anchor essays (N) increases.\n\n",
        "**Noise Model Sources:**\n",
        f"- LLM noise: {results.llm_noise_source}\n",
        f"- Teacher noise: {results.teacher_noise_source}\n\n",
        "Four 95% confidence intervals are shown, each isolating a different uncertainty source:\n",
        "- **CI (Sampling)**: Bootstrap resampling only (what if we had different essays?)\n",
        "- **CI (LLM)**: LLM stability noise only (how much does anchor order matter?)\n",
        "- **CI (Teacher)**: Teacher/GT noise only (how much do teachers disagree?)\n",
        "- **CI (All)**: All sources combined (sampling + LLM + teacher)\n\n",
        _format_scaling_table(results.scaling_results),
    ])

    if results.scaling_results:
        r = results.scaling_results[0]
        sections.append(f"\n_Leave-one-out evaluation on {r.n_targets} target essays._\n")
        sections.append("\n### QWK Chart\n")
        sections.append("![QWK Scores by Number of Anchors](qwk_chart.png)\n")

    # Add pairwise comparison section
    if results.pairwise_results:
        sections.extend([
            "\n---\n",
            "## Pairwise Comparison: P(N=X beats N=Y)\n",
            "This analysis applies the SAME scenario (bootstrap sample + noise) to all N values,\n",
            "then counts which N produces higher QWK in each scenario.\n\n",
            _format_pairwise_table(results.pairwise_results),
            "\n### Pairwise Heatmap\n",
            "![Pairwise Win Probabilities](pairwise_heatmap.png)\n",
        ])

    # Add Experiment 3: Content Stability section
    if results.content_stability_results:
        sections.extend([
            "\n---\n",
            "## Experiment 3: Content Stability\n",
            "Measures how the CHOICE of anchor essays (not just their order) affects predicted grades.\n",
            "Unlike Experiment 1 (same anchors, different orderings), this uses different anchor sets entirely.\n",
            "Values show probability ± SE of each deviation from the mode grade across different anchor sets.\n\n",
            _format_stability_table(results.content_stability_results),
        ])
        sections.append("\n### Content Stability Histograms\n")
        for sr in results.content_stability_results:
            sections.append(f"![Content Stability Histogram N={sr.n_anchors}](content_stability_histogram_n{sr.n_anchors}.png)\n")

    return "\n".join(sections)


def save_markdown_report(results: AnalysisResults, output_path: Path) -> None:
    """Save Markdown report to file."""
    report = generate_markdown_report(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def results_to_dict(results: AnalysisResults) -> dict[str, Any]:
    """Convert AnalysisResults to JSON-serializable dict."""
    result_dict = {
        "metadata": {
            "dataset_name": results.dataset_name,
            "model_name": results.model_name,
            "master_seed": results.master_seed,
            "timestamp": results.timestamp,
            "noise_sources": {
                "llm_noise": results.llm_noise_source,
                "teacher_noise": results.teacher_noise_source,
            },
        },
        "throughput_stats": results.throughput_stats,
        "experiment_1_stability": [
            {
                "n_anchors": r.n_anchors,
                "stability_vector": {str(k): v for k, v in r.stability_vector.items()},
                "standard_errors": {str(k): v for k, v in r.standard_errors.items()},
                "n_samples": r.n_samples,
            }
            for r in results.stability_results
        ],
        "experiment_2_scaling": [
            {
                "n_anchors": r.n_anchors,
                "raw_qwk": r.raw_qwk,
                "mean_qwk": r.mean_qwk,
                "ci_all": {"lower": r.lower_ci, "upper": r.upper_ci},
                "ci_sampling_only": {"lower": r.ci_sampling_only_lower, "upper": r.ci_sampling_only_upper},
                "ci_llm_only": {"lower": r.ci_llm_only_lower, "upper": r.ci_llm_only_upper},
                "ci_teacher_only": {"lower": r.ci_teacher_only_lower, "upper": r.ci_teacher_only_upper},
                "exact_accuracy": r.exact_accuracy,
                "near_accuracy": r.near_accuracy,
                "n_targets": r.n_targets,
            }
            for r in results.scaling_results
        ],
    }

    # Add pairwise comparison if available
    if results.pairwise_results:
        pr = results.pairwise_results
        result_dict["pairwise_comparison"] = {
            "n_values": pr.n_values,
            "n_iterations": pr.n_iterations,
            "win_matrix_qwk": {f"{k[0]}vs{k[1]}": v for k, v in pr.win_matrix_qwk.items()},
            "win_matrix_essay": {f"{k[0]}vs{k[1]}": v for k, v in pr.win_matrix_essay.items()},
            "avg_qwk_by_n": pr.avg_qwk_by_n,
        }

    # Add experiment 3 content stability if available
    if results.content_stability_results:
        result_dict["experiment_3_content_stability"] = [
            {
                "n_anchors": r.n_anchors,
                "stability_vector": {str(k): v for k, v in r.stability_vector.items()},
                "standard_errors": {str(k): v for k, v in r.standard_errors.items()},
                "n_samples": r.n_samples,
            }
            for r in results.content_stability_results
        ]

    return result_dict


def save_json_results(results: AnalysisResults, output_path: Path) -> None:
    """Save results as JSON."""
    data = results_to_dict(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
