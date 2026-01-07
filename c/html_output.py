"""Self-contained HTML output generation for analysis results."""

import base64
import gzip
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
from .core import (
    FullAnalysisResult,
    AnalysisResult,
    ComparisonResult,
    ComparisonAccuracyResult,
    GradesTableData,
)
from .html_template import generate_html_shell
from .html_components import generate_preact_app


def _compress_json(data: dict) -> str:
    """Compress JSON data with gzip and return base64-encoded string."""
    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    compressed = gzip.compress(json_str.encode('utf-8'), compresslevel=9)
    return base64.b64encode(compressed).decode('ascii')

# Load icon as base64 at module level
_ICON_BASE64: Optional[str] = None


def _get_icon_base64() -> str:
    """Load and cache the icon as base64."""
    global _ICON_BASE64
    if _ICON_BASE64 is None:
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            with open(icon_path, "rb") as f:
                _ICON_BASE64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            _ICON_BASE64 = ""
    return _ICON_BASE64


def generate_html(result: FullAnalysisResult, noise_assumption: str = "expected") -> str:
    """Generate a self-contained HTML report from analysis results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build dynamic title from EGF names
    if len(result.individual_results) == 1:
        page_title = result.individual_results[0].egf_name
    elif len(result.individual_results) > 1:
        names = [r.egf_name for r in result.individual_results]
        if len(names) <= 3:
            page_title = " vs ".join(names)
        else:
            page_title = f"{names[0]} + {len(names) - 1} others"
    else:
        page_title = "EGF Analysis"

    # Build favicon data URI
    icon_b64 = _get_icon_base64()
    favicon_link = f'<link rel="icon" type="image/png" href="data:image/png;base64,{icon_b64}">' if icon_b64 else ""

    # Build summary comment for IDE viewing
    summary_comment = ""
    if result.summary_markdown:
        summary_comment = f"<!--\nSUMMARY\n\n{result.summary_markdown}\n-->\n"

    # Generate static sections (SVG charts, individual analyses)
    static_content = _generate_static_content(result, noise_assumption, timestamp)

    # Build chunked app data and compress each chunk
    # Chunks are loaded progressively for faster initial render
    chunks = _build_chunked_app_data(result)
    compressed_chunks = {
        name: _compress_json(data) if data else ""
        for name, data in chunks.items()
    }

    # Generate CSS
    css = _generate_css()

    # Generate Preact app JS
    app_js = generate_preact_app()

    return generate_html_shell(
        page_title=page_title,
        favicon_link=favicon_link,
        css=css,
        summary_comment=summary_comment,
        static_content=static_content,
        compressed_chunks=compressed_chunks,
        app_js=app_js,
    )


def _generate_static_content(result: FullAnalysisResult, noise_assumption: str, timestamp: str) -> str:
    """Generate static HTML content (header, SVG charts, individual sections)."""
    parts = []

    # Header
    parts.append(f'''
        <header>
            <h1>EGF Analysis Report</h1>
            <div class="meta">Generated: {timestamp} | Noise assumption: {noise_assumption}</div>
        </header>
    ''')

    # Legend
    if result.legend:
        legend_items = [f"<li><strong>{k}</strong>: {v}</li>" for k, v in result.legend.items()]
        parts.append(f'<div class="legend"><h3>Legend</h3><ul>{"".join(legend_items)}</ul></div>')

    # QWK Overview
    qwk_chart_svg = generate_qwk_bar_chart(result)
    parts.append(f'''
        <section class="overview">
            <h2>QWK Overview</h2>
            <div class="chart">{qwk_chart_svg}</div>
        </section>
    ''')

    # Pairwise QWK Comparison
    if result.comparison:
        comparison_svg = generate_comparison_heatmap(result)
        parts.append(f'''
        <section class="comparison">
            <h2>Pairwise QWK Comparison</h2>
            <p>P(Row has higher QWK than Column) based on {result.comparison.n_iterations:,} bootstrap iterations using {result.comparison.n_common_essays} common essays.</p>
            <div class="chart">{comparison_svg}</div>
        </section>
        ''')

    # Comparison Accuracy Section (NEW)
    if result.comparison_accuracy:
        accuracy_section = _generate_comparison_accuracy_section(result)
        parts.append(accuracy_section)

    # Individual EGF sections
    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        parts.append(generate_individual_section(res, label))

    return "\n".join(parts)


def _generate_comparison_accuracy_section(result: FullAnalysisResult) -> str:
    """Generate the comparison accuracy overview section."""
    if not result.comparison_accuracy:
        return ""

    # Generate bar chart
    chart_svg = generate_comparison_accuracy_chart(result)

    # Generate NxN matrix if available
    matrix_section = ""
    if result.comparison_accuracy_matrix and len(result.comparison_accuracy) > 1:
        matrix_svg = generate_comparison_accuracy_heatmap(result)
        matrix_section = f'''
            <h3>Pairwise Accuracy Comparison</h3>
            <p>P(Row has higher comparison accuracy than Column)</p>
            <div class="chart">{matrix_svg}</div>
        '''

    # Build stats summary
    stats_items = []
    for egf_name, acc_result in result.comparison_accuracy.items():
        label = result.legend.get(egf_name, egf_name) if result.legend else egf_name
        # Find the label from legend by matching the name
        for lbl, name in (result.legend or {}).items():
            if name == egf_name:
                label = lbl
                break
        if not math.isnan(acc_result.raw_accuracy):
            stats_items.append(f"<li><strong>{label}</strong>: {acc_result.raw_accuracy:.1%} ({acc_result.n_comparisons} comparisons)</li>")

    return f'''
    <section class="comparison-accuracy">
        <h2>Comparison Accuracy</h2>
        <p>Accuracy of pairwise comparisons against ground truth grade ordering.
           Only comparisons with both GT grades are measured. GT ties are excluded from raw accuracy.
           For CI, comparisons where noised GT grades become equal are excluded from each bootstrap iteration.</p>
        <div class="chart">{chart_svg}</div>
        {matrix_section}
        <ul class="accuracy-stats">{" ".join(stats_items)}</ul>
    </section>
    '''


def _build_chunked_app_data(result: FullAnalysisResult) -> dict[str, dict]:
    """Build chunked JSON data for progressive loading.

    Returns dict with keys: 'core', 'submissions', 'grades', 'comparisons', 'llmCalls'
    Each chunk is loaded and merged progressively for faster initial render.
    """
    if not result.grades_table:
        return {'core': {}}

    grades_table = result.grades_table

    # CHUNK 1: Core metadata (tiny, loads first)
    core_data = {
        'egfNames': grades_table.egf_names,
        'egfLabels': grades_table.egf_labels,
        'maxGrade': grades_table.max_grade,
        'noiseAssumption': grades_table.noise_assumption,
    }

    # CHUNK 2: Submissions (without PDFs for now - PDFs loaded separately)
    submissions_data = {}
    pdfs_data = {}  # Separate heavy PDFs
    for sid, sub in grades_table.submissions.items():
        submissions_data[sid] = {
            'submission_id': sub.submission_id,
            'student_name': sub.student_name,
            'student_id': sub.student_id,
            'essay_markdown': sub.essay_markdown,
            'ground_truth_grade': sub.ground_truth_grade,
            'gt_distribution': sub.gt_distribution,
            'gt_justification': sub.gt_justification,
            # pdf_base64 moved to separate chunk
        }
        if sub.pdf_base64:
            pdfs_data[sid] = sub.pdf_base64

    # CHUNK 3: Grades (medium size)
    egf_grades_data = {}
    for egf_name, grades in grades_table.egf_grades.items():
        egf_grades_data[egf_name] = {}
        for sid, grade in grades.items():
            llm_calls_data = [
                {'call_id': call.call_id, 'pass_number': call.pass_number}
                for call in grade.llm_calls
            ]
            egf_grades_data[egf_name][sid] = {
                'submission_id': grade.submission_id,
                'grade': grade.grade,
                'grade_distribution': grade.grade_distribution,
                'justification': grade.justification,
                'llm_calls': llm_calls_data,
            }

    # CHUNK 4: Comparisons (medium size)
    egf_comparisons_data = {}
    for egf_name, comps_by_sub in grades_table.egf_comparisons.items():
        egf_comparisons_data[egf_name] = {}
        for sub_id, comps in comps_by_sub.items():
            egf_comparisons_data[egf_name][sub_id] = [
                {
                    'comparison_id': c.comparison_id,
                    'submission_a': c.submission_a,
                    'submission_b': c.submission_b,
                    'winner': c.winner,
                    'call_ids': c.call_ids,
                    'compared_at': c.compared_at,
                    'confidence': c.confidence,
                    'justification': c.justification,
                    'is_external': c.is_external,
                }
                for c in comps
            ]

    # CHUNK 5: LLM Calls (heaviest - loaded last)
    llm_calls_data = grades_table.all_llm_calls

    return {
        'core': core_data,
        'submissions': submissions_data,
        'grades': egf_grades_data,
        'comparisons': egf_comparisons_data,
        'llmCalls': llm_calls_data,
        'pdfs': pdfs_data,
    }


def generate_individual_section(result: AnalysisResult, label: str) -> str:
    """Generate HTML section for an individual EGF analysis."""
    qwk = result.qwk_result

    gt_mean, gt_lower, gt_upper = qwk.gt_noise_ci
    grading_mean, grading_lower, grading_upper = qwk.grading_noise_ci
    sampling_mean, sampling_lower, sampling_upper = qwk.sampling_ci
    combined_mean, combined_lower, combined_upper = qwk.combined_ci

    return f'''
    <section class="individual">
        <h2>{label}: {result.egf_name}</h2>
        {f'<p class="description">{result.grading_description}</p>' if result.grading_description else ''}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Raw QWK</div>
                <div class="value">{qwk.raw_qwk:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Exact Accuracy</div>
                <div class="value">{qwk.exact_accuracy:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Near Accuracy (±1)</div>
                <div class="value">{qwk.near_accuracy:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Essays</div>
                <div class="value">{qwk.n_essays}</div>
            </div>
        </div>

        <table class="ci-table">
            <thead>
                <tr>
                    <th>Variance Component</th>
                    <th>Mean QWK</th>
                    <th>95% CI</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>GT Noise Only</td>
                    <td>{gt_mean:.4f}</td>
                    <td>[{gt_lower:.3f}, {gt_upper:.3f}]</td>
                </tr>
                <tr>
                    <td>Grading Noise Only</td>
                    <td>{grading_mean:.4f}</td>
                    <td>[{grading_lower:.3f}, {grading_upper:.3f}]</td>
                </tr>
                <tr>
                    <td>Essay Sampling</td>
                    <td>{sampling_mean:.4f}</td>
                    <td>[{sampling_lower:.3f}, {sampling_upper:.3f}]</td>
                </tr>
                <tr>
                    <td><strong>Combined (All Noise)</strong></td>
                    <td><strong>{combined_mean:.4f}</strong></td>
                    <td><strong>[{combined_lower:.3f}, {combined_upper:.3f}]</strong></td>
                </tr>
            </tbody>
        </table>
    </section>
    '''


def generate_qwk_bar_chart(result: FullAnalysisResult) -> str:
    """Generate SVG bar chart showing QWK with 4 CI bars per file."""
    n_files = len(result.individual_results)
    if n_files == 0:
        return ""

    bar_width = 20
    group_width = bar_width * 4 + 40
    chart_width = max(400, n_files * group_width + 120)
    chart_height = 300
    margin_left = 50
    margin_bottom = 60
    margin_top = 30

    plot_height = chart_height - margin_top - margin_bottom

    colors = {
        'gt': '#ef4444',
        'grading': '#3b82f6',
        'sampling': '#22c55e',
        'combined': '#8b5cf6',
    }

    bars_svg = []

    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        qwk = res.qwk_result

        group_x = margin_left + idx * group_width + 20

        raw_y = margin_top + plot_height * (1 - qwk.raw_qwk)

        bars_svg.append(f'''
            <line x1="{group_x - 5}" y1="{raw_y}" x2="{group_x + bar_width * 4 + 10}" y2="{raw_y}"
                  stroke="#111" stroke-width="2" stroke-dasharray="4"/>
        ''')

        ci_data = [
            ('gt', qwk.gt_noise_ci),
            ('grading', qwk.grading_noise_ci),
            ('sampling', qwk.sampling_ci),
            ('combined', qwk.combined_ci),
        ]

        for bar_idx, (ci_type, (mean, lower, upper)) in enumerate(ci_data):
            if math.isnan(mean):
                continue

            bar_x = group_x + bar_idx * (bar_width + 5)
            mean_y = margin_top + plot_height * (1 - mean)
            lower_y = margin_top + plot_height * (1 - lower)
            upper_y = margin_top + plot_height * (1 - upper)
            color = colors[ci_type]

            bars_svg.append(f'''
                <line x1="{bar_x + bar_width/2}" y1="{lower_y}" x2="{bar_x + bar_width/2}" y2="{upper_y}"
                      stroke="{color}" stroke-width="3" opacity="0.5"/>
                <line x1="{bar_x + 3}" y1="{lower_y}" x2="{bar_x + bar_width - 3}" y2="{lower_y}"
                      stroke="{color}" stroke-width="2"/>
                <line x1="{bar_x + 3}" y1="{upper_y}" x2="{bar_x + bar_width - 3}" y2="{upper_y}"
                      stroke="{color}" stroke-width="2"/>
                <circle cx="{bar_x + bar_width/2}" cy="{mean_y}" r="4" fill="{color}"/>
            ''')

        bars_svg.append(f'''
            <text x="{group_x + bar_width * 2}" y="{chart_height - margin_bottom + 20}"
                  text-anchor="middle" font-size="12" font-weight="bold">{label}</text>
        ''')

    y_axis = []
    for i in range(6):
        val = i * 0.2
        y = margin_top + plot_height * (1 - val)
        y_axis.append(f'''
            <line x1="{margin_left - 5}" y1="{y}" x2="{chart_width - 20}" y2="{y}"
                  stroke="#e5e7eb" stroke-width="1"/>
            <text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="10">{val:.1f}</text>
        ''')

    legend = f'''
        <g transform="translate({chart_width - 150}, {margin_top})">
            <rect x="0" y="0" width="10" height="10" fill="{colors['gt']}"/>
            <text x="15" y="9" font-size="10">GT Noise</text>
            <rect x="0" y="15" width="10" height="10" fill="{colors['grading']}"/>
            <text x="15" y="24" font-size="10">Grading Noise</text>
            <rect x="0" y="30" width="10" height="10" fill="{colors['sampling']}"/>
            <text x="15" y="39" font-size="10">Sampling</text>
            <rect x="0" y="45" width="10" height="10" fill="{colors['combined']}"/>
            <text x="15" y="54" font-size="10">Combined</text>
            <line x1="0" y1="65" x2="20" y2="65" stroke="#111" stroke-width="2" stroke-dasharray="4"/>
            <text x="25" y="69" font-size="10">Raw QWK</text>
        </g>
    '''

    svg = f'''
    <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        {"".join(y_axis)}
        {"".join(bars_svg)}
        {legend}
        <text x="{margin_left - 35}" y="{chart_height / 2}" text-anchor="middle"
              font-size="12" transform="rotate(-90, {margin_left - 35}, {chart_height / 2})">QWK</text>
    </svg>
    '''

    return svg


def generate_comparison_heatmap(result: FullAnalysisResult) -> str:
    """Generate SVG heatmap for pairwise comparison."""
    if not result.comparison:
        return ""

    comp = result.comparison
    n = len(comp.egf_names)
    cell_size = 60
    margin = 50
    chart_size = n * cell_size + margin * 2

    cells = []

    for i in range(n):
        for j in range(n):
            x = margin + j * cell_size
            y = margin + i * cell_size

            if i == j:
                color = "#f3f4f6"
                text = "-"
            else:
                prob = comp.win_matrix.get((i, j), 0.5)
                # Three-point color scale: red (0) -> yellow (0.5) -> green (1)
                if prob >= 0.5:
                    t = (prob - 0.5) * 2
                    r = int(234 + (34 - 234) * t)
                    g = int(179 + (197 - 179) * t)
                    b = int(8 + (94 - 8) * t)
                else:
                    t = prob * 2
                    r = int(239 + (234 - 239) * t)
                    g = int(68 + (179 - 68) * t)
                    b = int(68 + (8 - 68) * t)
                color = f"rgb({r}, {g}, {b})"
                text = f"{prob:.0%}"

            cells.append(f'''
                <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}"
                      fill="{color}" stroke="#e5e7eb"/>
                <text x="{x + cell_size/2}" y="{y + cell_size/2 + 5}"
                      text-anchor="middle" font-size="12" font-weight="bold">{text}</text>
            ''')

    labels = result.labels[:n]
    row_labels = []
    col_labels = []
    for i, label in enumerate(labels):
        row_labels.append(f'''
            <text x="{margin - 10}" y="{margin + i * cell_size + cell_size/2 + 5}"
                  text-anchor="end" font-size="12" font-weight="bold">{label}</text>
        ''')
        col_labels.append(f'''
            <text x="{margin + i * cell_size + cell_size/2}" y="{margin - 10}"
                  text-anchor="middle" font-size="12" font-weight="bold">{label}</text>
        ''')

    svg = f'''
    <svg width="{chart_size}" height="{chart_size}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        {"".join(cells)}
        {"".join(row_labels)}
        {"".join(col_labels)}
    </svg>
    '''

    return svg


def generate_comparison_accuracy_chart(result: FullAnalysisResult) -> str:
    """Generate SVG bar chart showing comparison accuracy with CI."""
    if not result.comparison_accuracy:
        return ""

    n_files = len(result.comparison_accuracy)
    if n_files == 0:
        return ""

    bar_width = 40
    gap = 20
    chart_width = max(300, n_files * (bar_width + gap) + 100)
    chart_height = 200
    margin_left = 50
    margin_bottom = 40
    margin_top = 20

    plot_height = chart_height - margin_top - margin_bottom

    bars_svg = []
    idx = 0

    for egf_name, acc_result in result.comparison_accuracy.items():
        if math.isnan(acc_result.raw_accuracy):
            idx += 1
            continue

        # Find label
        label = egf_name
        for lbl, name in (result.legend or {}).items():
            if name == egf_name:
                label = lbl
                break

        mean, lower, upper = acc_result.accuracy_ci
        bar_x = margin_left + idx * (bar_width + gap)

        # Draw CI bar
        mean_y = margin_top + plot_height * (1 - mean)
        lower_y = margin_top + plot_height * (1 - max(0, lower))
        upper_y = margin_top + plot_height * (1 - min(1, upper))

        bars_svg.append(f'''
            <line x1="{bar_x + bar_width/2}" y1="{lower_y}" x2="{bar_x + bar_width/2}" y2="{upper_y}"
                  stroke="#8b5cf6" stroke-width="4" opacity="0.5"/>
            <line x1="{bar_x + 5}" y1="{lower_y}" x2="{bar_x + bar_width - 5}" y2="{lower_y}"
                  stroke="#8b5cf6" stroke-width="2"/>
            <line x1="{bar_x + 5}" y1="{upper_y}" x2="{bar_x + bar_width - 5}" y2="{upper_y}"
                  stroke="#8b5cf6" stroke-width="2"/>
            <circle cx="{bar_x + bar_width/2}" cy="{mean_y}" r="5" fill="#8b5cf6"/>
            <text x="{bar_x + bar_width/2}" y="{chart_height - 10}"
                  text-anchor="middle" font-size="12" font-weight="bold">{label}</text>
            <text x="{bar_x + bar_width/2}" y="{mean_y - 10}"
                  text-anchor="middle" font-size="10">{mean:.0%}</text>
        ''')
        idx += 1

    # Y-axis
    y_axis = []
    for i in range(6):
        val = i * 0.2
        y = margin_top + plot_height * (1 - val)
        y_axis.append(f'''
            <line x1="{margin_left - 5}" y1="{y}" x2="{chart_width - 20}" y2="{y}"
                  stroke="#e5e7eb" stroke-width="1"/>
            <text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="10">{val:.0%}</text>
        ''')

    svg = f'''
    <svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        {"".join(y_axis)}
        {"".join(bars_svg)}
        <text x="{margin_left - 35}" y="{chart_height / 2}" text-anchor="middle"
              font-size="12" transform="rotate(-90, {margin_left - 35}, {chart_height / 2})">Accuracy</text>
    </svg>
    '''

    return svg


def generate_comparison_accuracy_heatmap(result: FullAnalysisResult) -> str:
    """Generate SVG heatmap for pairwise comparison accuracy."""
    if not result.comparison_accuracy_matrix:
        return ""

    n = len(result.comparison_accuracy)
    cell_size = 60
    margin = 50
    chart_size = n * cell_size + margin * 2

    cells = []
    labels = []

    egf_names = list(result.comparison_accuracy.keys())

    for i in range(n):
        for j in range(n):
            x = margin + j * cell_size
            y = margin + i * cell_size

            if i == j:
                color = "#f3f4f6"
                text = "-"
            else:
                prob = result.comparison_accuracy_matrix.get((i, j), 0.5)
                if math.isnan(prob):
                    color = "#f3f4f6"
                    text = "-"
                else:
                    # Three-point color scale
                    if prob >= 0.5:
                        t = (prob - 0.5) * 2
                        r = int(234 + (34 - 234) * t)
                        g = int(179 + (197 - 179) * t)
                        b = int(8 + (94 - 8) * t)
                    else:
                        t = prob * 2
                        r = int(239 + (234 - 239) * t)
                        g = int(68 + (179 - 68) * t)
                        b = int(68 + (8 - 68) * t)
                    color = f"rgb({r}, {g}, {b})"
                    text = f"{prob:.0%}"

            cells.append(f'''
                <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}"
                      fill="{color}" stroke="#e5e7eb"/>
                <text x="{x + cell_size/2}" y="{y + cell_size/2 + 5}"
                      text-anchor="middle" font-size="12" font-weight="bold">{text}</text>
            ''')

    # Labels
    for i, egf_name in enumerate(egf_names):
        label = egf_name
        for lbl, name in (result.legend or {}).items():
            if name == egf_name:
                label = lbl
                break
        labels.append(f'''
            <text x="{margin - 10}" y="{margin + i * cell_size + cell_size/2 + 5}"
                  text-anchor="end" font-size="12" font-weight="bold">{label}</text>
            <text x="{margin + i * cell_size + cell_size/2}" y="{margin - 10}"
                  text-anchor="middle" font-size="12" font-weight="bold">{label}</text>
        ''')

    svg = f'''
    <svg width="{chart_size}" height="{chart_size}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        {"".join(cells)}
        {"".join(labels)}
    </svg>
    '''

    return svg


def _generate_css() -> str:
    """Generate all CSS styles for the report."""
    return '''
        :root {
            --primary: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        /* Loading spinner for progressive loading */
        .loading-spinner {
            width: 40px;
            height: 40px;
            margin: 0 auto;
            border: 3px solid var(--gray-200);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: var(--gray-50);
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        header h1 {
            color: var(--gray-900);
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }
        header .meta {
            color: var(--gray-600);
            font-size: 0.875rem;
        }
        .legend {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .legend h3 {
            font-size: 1rem;
            margin-bottom: 0.75rem;
            color: var(--gray-700);
        }
        .legend ul {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .legend li {
            font-size: 0.875rem;
            color: var(--gray-600);
        }
        section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        section h2 {
            color: var(--gray-900);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--gray-200);
        }
        section h3 {
            color: var(--gray-800);
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem 0;
        }
        .chart {
            display: flex;
            justify-content: center;
            padding: 1rem 0;
        }
        .chart svg {
            max-width: 100%;
            height: auto;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1rem;
        }
        .stat-card .label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gray-600);
            margin-bottom: 0.25rem;
        }
        .stat-card .value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--gray-900);
        }
        .ci-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        .ci-table th, .ci-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }
        .ci-table th {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gray-600);
            font-weight: 500;
        }
        .ci-table td {
            font-size: 0.875rem;
        }
        .accuracy-stats {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-top: 1rem;
        }
        .accuracy-stats li {
            font-size: 0.875rem;
            color: var(--gray-600);
        }
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-600);
            font-size: 0.875rem;
        }
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }
            header, section {
                padding: 1.5rem;
            }
        }
        .text-gray {
            color: var(--gray-500);
        }

        /* Grades Table */
        .grades-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .grades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        .grades-table th,
        .grades-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }
        .grades-table th {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gray-600);
            font-weight: 500;
            position: sticky;
            top: 0;
            background: white;
        }
        .grade-row {
            cursor: pointer;
            transition: background 0.15s;
        }
        .grade-row:hover {
            background: var(--gray-50);
        }
        .grade-cell {
            font-variant-numeric: tabular-nums;
        }
        .grade-match {
            color: var(--success);
        }
        .grade-close {
            color: var(--warning);
        }
        .grade-far {
            color: var(--danger);
        }
        .grade-neutral {
            background-color: var(--gray-100);
            color: var(--gray-500);
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal-content {
            background: white;
            border-radius: 12px;
            width: 95%;
            max-width: 1400px;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .modal-header h3 {
            margin: 0;
            font-size: 1.25rem;
            flex: 1;
        }
        .modal-back {
            background: none;
            border: none;
            font-size: 1.25rem;
            cursor: pointer;
            color: var(--gray-600);
            padding: 0.25rem 0.5rem;
            line-height: 1;
            border-radius: 4px;
        }
        .modal-back:hover {
            color: var(--gray-900);
            background: var(--gray-100);
        }
        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--gray-600);
            padding: 0.25rem 0.5rem;
            line-height: 1;
        }
        .modal-close:hover {
            color: var(--gray-900);
        }

        /* Tabs */
        .modal-tabs {
            display: flex;
            border-bottom: 1px solid var(--gray-200);
            padding: 0 1rem;
            overflow-x: auto;
            flex-shrink: 0;
        }
        .tab-btn {
            padding: 0.75rem 1rem;
            border: none;
            background: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            color: var(--gray-600);
            font-size: 0.875rem;
            white-space: nowrap;
        }
        .tab-btn:hover {
            color: var(--gray-900);
        }
        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }
        .modal-body {
            padding: 1.5rem;
            overflow-y: auto;
            flex: 1;
        }

        /* Markdown Content - Syntax Highlighted Source View */
        .markdown-content {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.875rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .markdown-content .md-header {
            color: var(--primary);
            font-weight: bold;
        }
        .markdown-content .md-bold {
            font-weight: bold;
        }
        .markdown-content .md-italic {
            font-style: italic;
        }
        .markdown-content .md-code {
            background: var(--gray-100);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
        }
        .markdown-content .md-code-block {
            display: block;
            background: var(--gray-100);
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        .markdown-content .md-link {
            color: var(--primary);
        }
        .markdown-content .md-list {
            color: var(--gray-500);
        }
        .markdown-content .md-blockquote {
            color: var(--gray-600);
            border-left: 3px solid var(--gray-300);
            padding-left: 0.75rem;
            display: block;
        }
        .markdown-content .md-hr {
            color: var(--gray-400);
        }

        /* Grade Display */
        .grade-display {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .grade-display .grade-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--gray-900);
        }
        .grade-display .grade-max {
            font-size: 1rem;
            color: var(--gray-500);
        }

        /* Histogram */
        .histogram-section {
            margin: 1.5rem 0;
        }
        .histogram-section h4 {
            font-size: 0.875rem;
            color: var(--gray-600);
            margin-bottom: 0.75rem;
        }
        .histogram-container {
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1rem;
        }

        /* Justification */
        .justification-section {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--gray-200);
        }
        .justification-section h4 {
            font-size: 0.875rem;
            color: var(--gray-600);
            margin-bottom: 0.75rem;
        }

        /* Table container */
        .table-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--gray-200);
            border-radius: 8px;
        }
        .table-container .grades-table th {
            background: var(--gray-50);
        }
        /* Virtual scrolling table */
        .table-container.virtual-table {
            max-height: none;  /* Height set inline */
        }
        .virtual-table .grades-table th {
            position: sticky;
            top: 0;
            z-index: 10;
            background: white;
            box-shadow: 0 1px 0 var(--gray-200);
        }

        /* LLM Calls Section */
        .llm-calls-section {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--gray-200);
        }
        .llm-calls-section h4 {
            font-size: 0.875rem;
            color: var(--gray-600);
            margin-bottom: 0.75rem;
        }

        /* Subtabs */
        .subtabs {
            display: flex;
            gap: 0.25rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .subtab-btn {
            padding: 0.4rem 0.75rem;
            border: 1px solid var(--gray-300);
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            color: var(--gray-600);
            transition: all 0.15s;
        }
        .subtab-btn:hover {
            background: var(--gray-50);
            border-color: var(--gray-400);
        }
        .subtab-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        /* Inspection Screen */
        .inspection-screen {
            margin-top: 1rem;
        }
        .inspection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .inspection-header h4 {
            font-size: 1rem;
            color: var(--gray-800);
        }
        .inspection-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: var(--gray-500);
        }
        .call-id {
            font-family: ui-monospace, monospace;
            color: var(--gray-500);
        }

        /* JSON display */
        .json-display {
            background: var(--gray-900);
            color: #e5e7eb;
            padding: 1rem;
            border-radius: 8px;
            font-family: ui-monospace, monospace;
            font-size: 0.75rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.5;
        }
        .json-display .json-key {
            color: #93c5fd;
        }
        .json-display .json-string {
            color: #86efac;
        }
        .json-display .json-number {
            color: #fcd34d;
        }
        .json-display .json-boolean {
            color: #f472b6;
        }
        .json-display .json-null {
            color: #9ca3af;
        }

        /* Collapsible sections */
        .collapsible-section {
            margin-top: 1rem;
            border: 1px solid var(--gray-200);
            border-radius: 8px;
            overflow: hidden;
        }
        .collapsible-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: var(--gray-50);
            cursor: pointer;
            user-select: none;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--gray-700);
            transition: background 0.15s;
        }
        .collapsible-header:hover {
            background: var(--gray-100);
        }
        .collapsible-header .chevron {
            transition: transform 0.2s;
            font-size: 0.75rem;
            color: var(--gray-500);
        }
        .collapsible-section.open .collapsible-header .chevron {
            transform: rotate(90deg);
        }
        .collapsible-content {
            display: none;
            padding: 1rem;
            border-top: 1px solid var(--gray-200);
        }
        .collapsible-section.open .collapsible-content {
            display: block;
        }

        /* LLM Input/Output sections */
        .llm-output-section, .llm-input-section {
            margin-bottom: 1rem;
        }
        .llm-output-section h5, .llm-input-section h5 {
            font-size: 0.8rem;
            color: var(--gray-600);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .llm-output-content {
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1rem;
            min-height: 100px;
            max-height: 500px;
            overflow-y: auto;
        }

        /* Markdown Source View */
        .markdown-source {
            background: var(--gray-900);
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 8px;
            font-family: ui-monospace, 'Cascadia Code', 'Fira Code', Consolas, monospace;
            font-size: 0.8rem;
            line-height: 1.6;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 500px;
            overflow-y: auto;
        }
        .markdown-source .md-header {
            color: #569cd6;
            font-weight: bold;
        }
        .markdown-source .md-bold {
            color: #ce9178;
            font-weight: bold;
        }
        .markdown-source .md-code {
            color: #d7ba7d;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        .markdown-source .md-code-block {
            color: #d7ba7d;
        }
        .markdown-source .md-link {
            color: #4ec9b0;
        }
        .markdown-source .md-list {
            color: #6a9955;
        }
        .markdown-source .md-blockquote {
            color: #608b4e;
        }

        /* PDF Viewer */
        .pdf-container {
            height: 70vh;
            min-height: 500px;
        }
        .pdf-viewer {
            width: 100%;
            height: 100%;
            border: 1px solid var(--gray-200);
            border-radius: 8px;
            background: var(--gray-100);
        }
        .pdf-not-available {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
        }

        /* Comparison Components */
        .comparison-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            border: 1px solid var(--gray-200);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
        }
        .comparison-card:hover {
            background: var(--gray-50);
            border-color: var(--gray-300);
        }
        .comparison-summary {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .comparison-summary .participant {
            font-weight: 500;
        }
        .comparison-summary .participant.winner {
            color: var(--success);
        }
        .comparison-summary .vs {
            color: var(--gray-400);
            font-size: 0.75rem;
        }
        .comparison-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .comparison-meta .confidence {
            font-size: 0.75rem;
            color: var(--gray-500);
        }
        .external-badge {
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            background: var(--gray-200);
            color: var(--gray-600);
            border-radius: 4px;
        }
        .winner-badge {
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            background: var(--success);
            color: white;
            border-radius: 4px;
        }
        .tie-badge {
            font-size: 0.875rem;
            padding: 0.25rem 0.75rem;
            background: var(--gray-200);
            color: var(--gray-700);
            border-radius: 4px;
        }
        .comparison-pair {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }
        .submission-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem 2rem;
            border: 2px solid var(--gray-200);
            border-radius: 8px;
            min-width: 150px;
        }
        .submission-card.winner {
            border-color: var(--success);
            background: rgba(34, 197, 94, 0.05);
        }
        .submission-card.external {
            border-style: dashed;
        }
        .submission-card .submission-label {
            font-size: 0.75rem;
            color: var(--gray-500);
            margin-bottom: 0.25rem;
        }
        .submission-card .submission-name {
            font-weight: 600;
            color: var(--gray-800);
        }
        .vs-large {
            font-size: 1.25rem;
            color: var(--gray-400);
            font-weight: bold;
        }
        .result-display {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .winner-announcement {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--success);
        }
        .back-btn {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--gray-300);
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            color: var(--gray-600);
            transition: all 0.15s;
        }
        .back-btn:hover {
            background: var(--gray-50);
            border-color: var(--gray-400);
        }
        .comparisons-list {
            margin-top: 1rem;
        }
        .comparisons-count {
            font-size: 0.875rem;
            color: var(--gray-500);
            margin-bottom: 1rem;
        }
        .no-comparisons {
            font-style: italic;
        }

        /* Comparison Table Cells */
        .comparison-sub-cell {
            cursor: pointer;
            color: var(--primary);
        }
        .comparison-sub-cell:hover {
            text-decoration: underline;
        }

        /* Comparison Pair Modal */
        .comparison-pair-header {
            display: flex;
            justify-content: center;
            align-items: stretch;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .comparison-essay-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1.25rem 2rem;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            min-width: 180px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .comparison-essay-card:hover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.05);
        }
        .comparison-essay-card .essay-label {
            font-size: 0.75rem;
            color: var(--gray-500);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }
        .comparison-essay-card .essay-name {
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--gray-800);
            margin-bottom: 0.5rem;
        }
        .comparison-essay-card .essay-name.clickable {
            color: var(--primary);
        }
        .comparison-essay-card .essay-grade {
            font-size: 0.875rem;
            color: var(--gray-600);
        }
        .comparison-essay-card .gt-winner-badge {
            margin-top: 0.5rem;
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            background: var(--success);
            color: white;
            border-radius: 4px;
        }
        .vs-divider {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.25rem;
        }
        .vs-divider .vs-text {
            font-size: 1.25rem;
            font-weight: bold;
            color: var(--gray-400);
        }
        .vs-divider .gap-text {
            font-size: 0.75rem;
            color: var(--gray-500);
        }
        .gt-tie-notice {
            text-align: center;
            padding: 0.75rem;
            background: var(--gray-100);
            border-radius: 8px;
            color: var(--gray-600);
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }
        .egf-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--gray-200);
        }
        .comparison-result-section {
            margin-top: 1rem;
        }
        .comparison-result-section h4 {
            font-size: 1rem;
            color: var(--gray-800);
            margin-bottom: 1rem;
        }
        .result-summary {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        .result-summary .result-label {
            color: var(--gray-600);
        }
        .result-summary .result-value {
            font-weight: 600;
            color: var(--gray-900);
        }
        .result-summary .result-value.correct {
            color: var(--success);
        }
        .result-summary .result-value.incorrect {
            color: var(--danger);
        }
        .result-summary .check-mark {
            color: var(--success);
        }
        .result-summary .cross-mark {
            color: var(--danger);
        }
        .result-summary .result-confidence {
            font-size: 0.875rem;
            color: var(--gray-500);
        }
    '''


def save_html_report(result: FullAnalysisResult, output_path: Path, noise_assumption: str = "expected") -> None:
    """Save analysis results as a self-contained HTML file."""
    html = generate_html(result, noise_assumption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
