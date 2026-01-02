"""Visualization generation for experiment analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .report import StabilityResult, ScalingResult, ScalingResultByAssumption, PairwiseComparisonResult


def save_stability_histograms(
    results: list[StabilityResult],
    output_dir: Path,
    filename_prefix: str = "stability_histogram",
) -> None:
    """Generate and save deviation histogram images for each N value.

    Creates one histogram per N value showing the distribution of
    deviations from the mode grade.

    Args:
        results: List of stability results to visualize
        output_dir: Directory to save images
        filename_prefix: Prefix for output filenames (default: "stability_histogram")
    """
    # Determine title based on prefix
    is_content = "content" in filename_prefix.lower()
    title_type = "Content Stability" if is_content else "Permutation Stability"
    bar_color = "#9b59b6" if is_content else "#4a90d9"  # Purple for content, blue for permutation
    edge_color = "#7d3c98" if is_content else "#2c5aa0"

    for result in results:
        fig, ax = plt.subplots(figsize=(8, 5))

        deviations = sorted(result.stability_vector.keys())
        probs = [result.stability_vector[d] for d in deviations]
        errors = [result.standard_errors.get(d, 0) for d in deviations]

        x_labels = [f"{d:+d}" for d in deviations]
        x_positions = np.arange(len(deviations))

        bars = ax.bar(
            x_positions,
            probs,
            yerr=errors,
            capsize=4,
            color=bar_color,
            edgecolor=edge_color,
            linewidth=1,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

        ax.set_xlabel("Deviation from Mode", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(f"{title_type} Distribution (N={result.n_anchors})", fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, prob, err in zip(bars, probs, errors):
            height = bar.get_height()
            ax.annotate(
                f"{prob:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height + err + 0.02),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        output_path = output_dir / f"{filename_prefix}_n{result.n_anchors}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved histogram: {output_path.name}")


def save_qwk_chart(
    results: list[ScalingResult],
    output_dir: Path,
    llm_noise_source: str = "Experiment 1 data",
    teacher_noise_source: str = "Source EDF file",
) -> None:
    """Generate and save QWK charts with confidence intervals.

    Creates four separate plots showing raw QWK and mean QWK with CIs:
    - Sampling noise only (bootstrap resampling, no noise models)
    - LLM noise only (stability noise, no resampling)
    - Teacher noise only (GT noise, no resampling)
    - All noise sources combined (resampling + LLM + teacher)
    """
    n_values = [r.n_anchors for r in results]
    raw_qwk = [r.raw_qwk for r in results]
    mean_qwk = [r.mean_qwk for r in results]

    # Shorten source names for display
    def shorten_source(src: str) -> str:
        if "DIPARATIVE_N20" in src:
            return "DIPARATIVE_N20_NOISE_MODEL"
        if "Experiment 1" in src:
            return "Experiment 1 data"
        if "EDF" in src:
            return "EDF teacher noise"
        if "CLI override" in src:
            return "CLI override"
        if "Default" in src:
            return "Default noise model"
        return src

    llm_short = shorten_source(llm_noise_source)
    teacher_short = shorten_source(teacher_noise_source)

    # Check if we have separate CIs
    has_separate_cis = hasattr(results[0], "ci_sampling_only_lower") and results[0].ci_sampling_only_lower is not None

    if has_separate_cis:
        # Create 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axes = axes.flatten()  # Flatten for easier iteration

        ci_configs = [
            {
                "title": "Sampling Only\n(Bootstrap resampling, no noise)",
                "lower": [r.ci_sampling_only_lower for r in results],
                "upper": [r.ci_sampling_only_upper for r in results],
                "color": "#748ffc",
            },
            {
                "title": f"LLM Noise Only\n(Source: {llm_short})",
                "lower": [r.ci_llm_only_lower for r in results],
                "upper": [r.ci_llm_only_upper for r in results],
                "color": "#51cf66",
            },
            {
                "title": f"Teacher Noise Only\n(Source: {teacher_short})",
                "lower": [r.ci_teacher_only_lower for r in results],
                "upper": [r.ci_teacher_only_upper for r in results],
                "color": "#ffa94d",
            },
            {
                "title": "All Noise Sources\n(Sampling + LLM + Teacher)",
                "lower": [r.lower_ci for r in results],
                "upper": [r.upper_ci for r in results],
                "color": "#ff6b6b",
            },
        ]

        for ax, cfg in zip(axes, ci_configs):
            # Plot raw QWK as hollow circles
            ax.scatter(
                n_values,
                raw_qwk,
                color="#4a90d9",
                s=80,
                marker="o",
                facecolors="none",
                linewidths=2,
                label="Raw QWK",
                zorder=3,
            )

            # Plot mean QWK with CI error bars
            lower_err = [max(0, m - l) for m, l in zip(mean_qwk, cfg["lower"])]
            upper_err = [max(0, u - m) for m, u in zip(mean_qwk, cfg["upper"])]

            ax.errorbar(
                n_values,
                mean_qwk,
                yerr=[lower_err, upper_err],
                fmt="o",
                color=cfg["color"],
                ecolor=cfg["color"],
                elinewidth=2,
                capsize=5,
                capthick=2,
                markersize=8,
                label="Mean QWK (95% CI)",
                zorder=2,
            )

            ax.set_xlabel("Number of Anchors (N)", fontsize=11)
            ax.set_title(cfg["title"], fontsize=11)
            ax.set_xticks(n_values)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="lower right", fontsize=9)

        # Set y-labels for left column only
        axes[0].set_ylabel("QWK Score", fontsize=11)
        axes[2].set_ylabel("QWK Score", fontsize=11)

        plt.suptitle("QWK Scores by Number of Anchor Essays", fontsize=13, y=1.02)
        plt.tight_layout()
    else:
        # Single plot fallback
        fig, ax = plt.subplots(figsize=(8, 5))

        lower_err = [max(0, m - r.lower_ci) for m, r in zip(mean_qwk, results)]
        upper_err = [max(0, r.upper_ci - m) for m, r in zip(mean_qwk, results)]

        ax.scatter(
            n_values,
            raw_qwk,
            color="#4a90d9",
            s=80,
            marker="o",
            facecolors="none",
            linewidths=2,
            label="Raw QWK",
            zorder=3,
        )

        ax.errorbar(
            n_values,
            mean_qwk,
            yerr=[lower_err, upper_err],
            fmt="o",
            color="#ff6b6b",
            ecolor="#ff6b6b",
            elinewidth=2,
            capsize=5,
            capthick=2,
            markersize=8,
            label="Mean QWK (95% CI)",
            zorder=2,
        )

        ax.set_xlabel("Number of Anchors (N)", fontsize=11)
        ax.set_ylabel("QWK Score", fontsize=11)
        ax.set_title("QWK Scores by Number of Anchor Essays", fontsize=12)
        ax.set_xticks(n_values)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()

    output_path = output_dir / "qwk_chart.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved QWK chart: {output_path.name}")


def _build_heatmap_matrix(
    n_values: list[int],
    win_matrix: dict[tuple[int, int], float],
) -> np.ndarray:
    """Build a numpy matrix from a win_matrix dict."""
    n = len(n_values)
    matrix = np.zeros((n, n))
    for i, n_i in enumerate(n_values):
        for j, n_j in enumerate(n_values):
            if i == j:
                matrix[i, j] = 0.5
            else:
                matrix[i, j] = win_matrix.get((n_i, n_j), 0)
    return matrix


def save_pairwise_heatmap(
    result: PairwiseComparisonResult,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Generate and save heatmaps of pairwise win probabilities.

    Creates two heatmaps: one for QWK-based, one for essay-level comparison.
    Green = high probability, Red = low probability.

    Args:
        result: Pairwise comparison result
        output_dir: Directory to save the heatmap
        filename_suffix: Optional suffix for filename (e.g., "optimistic")
    """
    n_values = result.n_values
    n = len(n_values)

    # Use labels if available, otherwise fall back to N values
    if result.labels and len(result.labels) == n:
        display_labels = result.labels
    else:
        display_labels = [f"N={nv}" for nv in n_values]

    # Build matrices
    matrix_qwk = _build_heatmap_matrix(n_values, result.win_matrix_qwk)
    matrix_essay = _build_heatmap_matrix(n_values, result.win_matrix_essay)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = [
        "QWK-Based: P(Row has higher QWK)",
        "Essay-Level: P(Row equal or closer to GT)",
    ]
    matrices = [matrix_qwk, matrix_essay]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("P(Row beats Column)", rotation=-90, va="bottom", fontsize=9)

        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_labels, fontsize=10, fontweight="bold")
        ax.set_yticklabels(display_labels, fontsize=10, fontweight="bold")

        # Add text annotations
        for i in range(n):
            for j in range(n):
                if i == j:
                    text = "-"
                else:
                    text = f"{matrix[i, j]:.0%}"
                color = "white" if matrix[i, j] < 0.3 or matrix[i, j] > 0.7 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Competitor", fontsize=10)
        ax.set_ylabel("Challenger", fontsize=10)

    # Add assumption to title if suffix provided
    title_suffix = f" - {filename_suffix.capitalize()}" if filename_suffix else ""
    plt.suptitle(f"Pairwise Win Probabilities{title_suffix} ({result.n_iterations:,} iterations)", fontsize=12, y=1.02)
    plt.tight_layout()

    # Use suffix in filename
    filename = f"pairwise_heatmap_{filename_suffix}.png" if filename_suffix else "pairwise_heatmap.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pairwise heatmap: {output_path.name}")


def save_multi_assumption_qwk_chart(
    results: list[ScalingResultByAssumption],
    output_dir: Path,
    ci_type: str = "combined",
) -> None:
    """Generate QWK chart comparing all 3 noise assumptions.

    Creates a single plot with 3 lines (one per assumption) showing how
    QWK varies across files under different noise assumptions.

    Args:
        results: List of results for each file
        output_dir: Directory to save the chart
        ci_type: Which CI type to display ("sampling_only", "llm_only",
                 "teacher_only", "combined")
    """
    # Use labels if available, otherwise use indices
    x_labels = [r.label if r.label else str(i) for i, r in enumerate(results)]
    x_positions = list(range(len(results)))
    raw_qwk = [r.raw_qwk for r in results]

    # Colors for each assumption
    assumption_colors = {
        "optimistic": "#51cf66",  # Green
        "expected": "#748ffc",    # Blue
        "pessimistic": "#ff6b6b", # Red
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw QWK as hollow circles
    ax.scatter(
        x_positions,
        raw_qwk,
        color="#333333",
        s=100,
        marker="o",
        facecolors="none",
        linewidths=2,
        label="Raw QWK",
        zorder=4,
    )

    # Plot each assumption
    for assumption in ["optimistic", "expected", "pessimistic"]:
        means = []
        lowers = []
        uppers = []

        for r in results:
            if assumption in r.by_assumption and ci_type in r.by_assumption[assumption]:
                mean, lower, upper = r.by_assumption[assumption][ci_type]
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
            else:
                means.append(r.raw_qwk)
                lowers.append(r.raw_qwk)
                uppers.append(r.raw_qwk)

        color = assumption_colors[assumption]

        # Plot error bars
        lower_err = [max(0, m - l) for m, l in zip(means, lowers)]
        upper_err = [max(0, u - m) for m, u in zip(means, uppers)]

        ax.errorbar(
            x_positions,
            means,
            yerr=[lower_err, upper_err],
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,
            capthick=2,
            markersize=8,
            label=f"{assumption.capitalize()} (95% CI)",
            zorder=3,
            alpha=0.8,
        )

    ax.set_xlabel("File", fontsize=12)
    ax.set_ylabel("QWK Score", fontsize=12)
    ax.set_title("QWK by Noise Assumption", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "qwk_by_assumption.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved multi-assumption QWK chart: {output_path.name}")


def save_multi_assumption_grid_chart(
    results: list[ScalingResultByAssumption],
    output_dir: Path,
) -> None:
    """Generate 4-panel grid showing QWK by CI type and assumption.

    Creates a 2x2 grid:
    - Top-left: Sampling only
    - Top-right: LLM noise only
    - Bottom-left: Teacher noise only
    - Bottom-right: All combined
    """
    # Use labels if available, otherwise use indices
    x_labels = [r.label if r.label else str(i) for i, r in enumerate(results)]
    x_positions = list(range(len(results)))
    raw_qwk = [r.raw_qwk for r in results]

    assumption_colors = {
        "optimistic": "#51cf66",
        "expected": "#748ffc",
        "pessimistic": "#ff6b6b",
    }

    ci_types = [
        ("sampling_only", "Sampling Only"),
        ("llm_only", "LLM Noise Only"),
        ("teacher_only", "Teacher Noise Only"),
        ("combined", "All Combined"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for ax, (ci_type, ci_title) in zip(axes, ci_types):
        # Plot raw QWK
        ax.scatter(
            x_positions,
            raw_qwk,
            color="#333333",
            s=60,
            marker="o",
            facecolors="none",
            linewidths=1.5,
            label="Raw QWK",
            zorder=4,
        )

        # Plot each assumption
        for assumption in ["optimistic", "expected", "pessimistic"]:
            means = []
            lowers = []
            uppers = []

            for r in results:
                if assumption in r.by_assumption and ci_type in r.by_assumption[assumption]:
                    mean, lower, upper = r.by_assumption[assumption][ci_type]
                    means.append(mean)
                    lowers.append(lower)
                    uppers.append(upper)
                else:
                    means.append(r.raw_qwk)
                    lowers.append(r.raw_qwk)
                    uppers.append(r.raw_qwk)

            color = assumption_colors[assumption]
            lower_err = [max(0, m - l) for m, l in zip(means, lowers)]
            upper_err = [max(0, u - m) for m, u in zip(means, uppers)]

            ax.errorbar(
                x_positions,
                means,
                yerr=[lower_err, upper_err],
                fmt="o-",
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                markersize=6,
                label=f"{assumption.capitalize()}",
                zorder=3,
                alpha=0.8,
            )

        ax.set_title(ci_title, fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    # Set common labels
    axes[0].set_ylabel("QWK Score", fontsize=11)
    axes[2].set_ylabel("QWK Score", fontsize=11)
    axes[2].set_xlabel("File", fontsize=11)
    axes[3].set_xlabel("File", fontsize=11)

    # Add legend to first subplot
    axes[0].legend(loc="lower right", fontsize=8)

    plt.suptitle("QWK Scores by CI Type and Noise Assumption", fontsize=13, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "qwk_grid_by_assumption.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved multi-assumption grid chart: {output_path.name}")
