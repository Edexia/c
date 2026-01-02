"""Self-contained HTML output generation for analysis results."""

import base64
import io
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
from .core import FullAnalysisResult, AnalysisResult, ComparisonResult, QWKResult, GradesTableData


def generate_html(result: FullAnalysisResult, noise_assumption: str = "expected") -> str:
    """Generate a self-contained HTML report from analysis results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build summary comment for IDE viewing
    summary_comment = ""
    if result.summary_markdown:
        summary_comment = f"<!--\nSUMMARY\n\n{result.summary_markdown}\n-->\n"

    qwk_chart_svg = generate_qwk_bar_chart(result)
    comparison_svg = ""
    if result.comparison:
        comparison_svg = generate_comparison_heatmap(result)

    individual_sections = []
    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        individual_sections.append(generate_individual_section(res, label))

    legend_html = ""
    if result.legend:
        legend_items = [f"<li><strong>{k}</strong>: {v}</li>" for k, v in result.legend.items()]
        legend_html = f'<div class="legend"><h3>Legend</h3><ul>{"".join(legend_items)}</ul></div>'

    comparison_section = ""
    if result.comparison:
        comparison_section = f'''
        <section class="comparison">
            <h2>Pairwise Comparison</h2>
            <p>P(Row has higher QWK than Column) based on {result.comparison.n_iterations:,} bootstrap iterations using {result.comparison.n_common_essays} common essays.</p>
            <div class="chart">{comparison_svg}</div>
        </section>
        '''

    # Generate grades table section and modal if available
    grades_table_section = ""
    grades_table_css = ""
    grades_table_modal = ""
    grades_table_js = ""
    if result.grades_table:
        grades_table_css = generate_grades_table_css()
        grades_table_section = generate_grades_table_section(result.grades_table)
        grades_table_modal = generate_modal_html(result.grades_table)
        grades_table_js = generate_grades_table_js(result.grades_table)

    html = f'''{summary_comment}<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EGF Analysis Report</title>
    <style>
        :root {{
            --primary: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: var(--gray-50);
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        header h1 {{
            color: var(--gray-900);
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}
        header .meta {{
            color: var(--gray-600);
            font-size: 0.875rem;
        }}
        .legend {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            font-size: 1rem;
            margin-bottom: 0.75rem;
            color: var(--gray-700);
        }}
        .legend ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        .legend li {{
            font-size: 0.875rem;
            color: var(--gray-600);
        }}
        section {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        section h2 {{
            color: var(--gray-900);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--gray-200);
        }}
        .chart {{
            display: flex;
            justify-content: center;
            padding: 1rem 0;
        }}
        .chart svg {{
            max-width: 100%;
            height: auto;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .stat-card {{
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1rem;
        }}
        .stat-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gray-600);
            margin-bottom: 0.25rem;
        }}
        .stat-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--gray-900);
        }}
        .stat-card .subvalue {{
            font-size: 0.75rem;
            color: var(--gray-600);
        }}
        .ci-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        .ci-table th, .ci-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }}
        .ci-table th {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gray-600);
            font-weight: 500;
        }}
        .ci-table td {{
            font-size: 0.875rem;
        }}
        .ci-bar {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .ci-bar .bar {{
            height: 8px;
            background: var(--primary);
            border-radius: 4px;
            opacity: 0.3;
        }}
        .ci-bar .point {{
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--gray-600);
            font-size: 0.875rem;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            header, section {{
                padding: 1.5rem;
            }}
        }}
        .text-gray {{
            color: var(--gray-500);
        }}
        {grades_table_css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>EGF Analysis Report</h1>
            <div class="meta">Generated: {timestamp} | Noise assumption: {noise_assumption}</div>
        </header>

        {legend_html}

        <section class="overview">
            <h2>QWK Overview</h2>
            <div class="chart">{qwk_chart_svg}</div>
        </section>

        {comparison_section}

        {grades_table_section}

        {"".join(individual_sections)}

        <footer>
            Generated by c - EGF Analysis CLI
        </footer>
    </div>
    {grades_table_modal}
    {grades_table_js}
</body>
</html>'''

    return html


def generate_individual_section(result: AnalysisResult, label: str) -> str:
    """Generate HTML section for an individual EGF analysis."""
    qwk = result.qwk_result

    gt_mean, gt_lower, gt_upper = qwk.gt_noise_ci
    grading_mean, grading_lower, grading_upper = qwk.grading_noise_ci
    sampling_mean, sampling_lower, sampling_upper = qwk.sampling_ci
    combined_mean, combined_lower, combined_upper = qwk.combined_ci

    def format_ci(mean: float, lower: float, upper: float) -> str:
        if math.isnan(mean):
            return "N/A"
        return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"

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
                if prob > 0.5:
                    intensity = (prob - 0.5) * 2
                    r = int(34 + (34 - 34) * intensity)
                    g = int(197 + (197 - 197) * intensity)
                    b = int(94 + (94 - 34) * intensity)
                    color = f"rgb({r}, {g}, {b})"
                else:
                    intensity = (0.5 - prob) * 2
                    r = int(239 + (239 - 239) * intensity)
                    g = int(68 + (68 - 68) * intensity)
                    b = int(68 + (68 - 68) * intensity)
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


def generate_grades_table_css() -> str:
    """Generate CSS for the grades table and modal."""
    return '''
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
        .grades-table tbody {
            max-height: 500px;
            overflow-y: auto;
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
            width: 90%;
            max-width: 900px;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        .modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h3 {
            margin: 0;
            font-size: 1.25rem;
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
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }

        /* Markdown Content */
        .markdown-content {
            line-height: 1.8;
        }
        .markdown-content h1 {
            font-size: 1.5rem;
            margin: 1.5rem 0 0.75rem 0;
            border-bottom: 1px solid var(--gray-200);
            padding-bottom: 0.5rem;
        }
        .markdown-content h2 {
            font-size: 1.25rem;
            margin: 1.25rem 0 0.5rem 0;
        }
        .markdown-content h3 {
            font-size: 1.1rem;
            margin: 1rem 0 0.5rem 0;
        }
        .markdown-content p {
            margin-bottom: 1rem;
        }
        .markdown-content ul, .markdown-content ol {
            margin: 0 0 1rem 1.5rem;
        }
        .markdown-content li {
            margin-bottom: 0.25rem;
        }
        .markdown-content code {
            background: var(--gray-100);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: ui-monospace, monospace;
            font-size: 0.875em;
        }
        .markdown-content pre {
            background: var(--gray-100);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 1rem;
        }
        .markdown-content pre code {
            background: none;
            padding: 0;
        }
        .markdown-content blockquote {
            border-left: 4px solid var(--gray-300);
            padding-left: 1rem;
            margin: 1rem 0;
            color: var(--gray-600);
        }

        /* Grade Display in Modal */
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

        /* Scrollable table container */
        .table-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--gray-200);
            border-radius: 8px;
        }
        .table-container .grades-table th {
            background: var(--gray-50);
        }
    '''


def generate_grades_table_js(grades_table: GradesTableData) -> str:
    """Generate JavaScript for the grades table interactivity."""
    # Convert dataclasses to JSON-serializable dicts
    submissions_data = {}
    for sid, sub in grades_table.submissions.items():
        submissions_data[sid] = {
            'submission_id': sub.submission_id,
            'student_name': sub.student_name,
            'student_id': sub.student_id,
            'essay_markdown': sub.essay_markdown,
            'ground_truth_grade': sub.ground_truth_grade,
            'gt_distribution': sub.gt_distribution,
        }

    egf_grades_data = {}
    for egf_name, grades in grades_table.egf_grades.items():
        egf_grades_data[egf_name] = {}
        for sid, grade in grades.items():
            egf_grades_data[egf_name][sid] = {
                'submission_id': grade.submission_id,
                'grade': grade.grade,
                'grade_distribution': grade.grade_distribution,
                'justification': grade.justification,
            }

    data = {
        'submissions': submissions_data,
        'egfGrades': egf_grades_data,
        'egfNames': grades_table.egf_names,
        'egfLabels': grades_table.egf_labels,
        'maxGrade': grades_table.max_grade,
        'noiseAssumption': grades_table.noise_assumption,
    }

    return f'''
<script id="gradesData" type="application/json">
{json.dumps(data)}
</script>
<script>
(function() {{
    const gradesData = JSON.parse(document.getElementById('gradesData').textContent);

    // Simple Markdown Parser
    function parseMarkdown(md) {{
        if (!md) return '<em class="text-gray">No content available</em>';

        let html = md
            // Escape HTML first
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            // Code blocks (before other processing)
            .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Headers
            .replace(/^### (.*)$/gm, '<h3>$1</h3>')
            .replace(/^## (.*)$/gm, '<h2>$1</h2>')
            .replace(/^# (.*)$/gm, '<h1>$1</h1>')
            // Bold and Italic
            .replace(/\\*\\*\\*(.+?)\\*\\*\\*/g, '<strong><em>$1</em></strong>')
            .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
            .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
            .replace(/___(.+?)___/g, '<strong><em>$1</em></strong>')
            .replace(/__(.+?)__/g, '<strong>$1</strong>')
            .replace(/_(.+?)_/g, '<em>$1</em>')
            // Blockquotes
            .replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>')
            // Unordered lists
            .replace(/^[\\*\\-] (.*)$/gm, '<li>$1</li>')
            // Ordered lists
            .replace(/^\\d+\\. (.*)$/gm, '<li>$1</li>')
            // Line breaks
            .replace(/\\n\\n/g, '</p><p>')
            .replace(/\\n/g, '<br>');

        // Wrap in paragraphs if not already wrapped
        if (!html.startsWith('<')) {{
            html = '<p>' + html + '</p>';
        }}

        // Clean up consecutive blockquotes
        html = html.replace(/<\\/blockquote>\\s*<blockquote>/g, '<br>');

        // Wrap lists
        html = html.replace(/(<li>.*<\\/li>)/gs, '<ul>$1</ul>');
        html = html.replace(/<\\/ul>\\s*<ul>/g, '');

        return html;
    }}

    // Generate histogram SVG
    function generateHistogramSVG(distribution, maxGrade, width, height) {{
        width = width || 300;
        height = height || 100;

        if (!distribution || distribution.length === 0) {{
            return '<em class="text-gray">No distribution data</em>';
        }}

        const barWidth = width / distribution.length;
        const maxProb = Math.max(...distribution) || 1;
        const chartHeight = height - 25;

        let bars = '';
        distribution.forEach((prob, i) => {{
            const barHeight = (prob / maxProb) * chartHeight;
            const x = i * barWidth;
            const y = chartHeight - barHeight;
            const opacity = prob > 0 ? 0.7 + (prob / maxProb) * 0.3 : 0.1;
            bars += `<rect x="${{x}}" y="${{y}}" width="${{barWidth - 1}}" height="${{barHeight}}" fill="#3b82f6" opacity="${{opacity}}"/>`;
            if (distribution.length <= 20) {{
                bars += `<text x="${{x + barWidth/2}}" y="${{height - 5}}" text-anchor="middle" font-size="9" fill="#6b7280">${{i}}</text>`;
            }}
        }});

        return `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f9fafb" rx="4"/>
            ${{bars}}
        </svg>`;
    }}

    // Open modal
    function openModal(submissionId) {{
        const submission = gradesData.submissions[submissionId];
        if (!submission) return;

        // Set title
        const title = submission.student_name || submission.student_id || submissionId;
        document.getElementById('modalTitle').textContent = title;

        // Populate Essay tab
        document.getElementById('essayContent').innerHTML = parseMarkdown(submission.essay_markdown);

        // Populate Ground Truth tab
        const gtGrade = submission.ground_truth_grade;
        document.getElementById('gtGradeValue').textContent = gtGrade !== null ? gtGrade : 'N/A';
        document.getElementById('gtGradeMax').textContent = '/ ' + gradesData.maxGrade;

        const gtHistContainer = document.getElementById('gtHistogram');
        if (submission.gt_distribution) {{
            document.getElementById('gtNoiseLabel').textContent = gradesData.noiseAssumption + ' noise';
            gtHistContainer.innerHTML = generateHistogramSVG(submission.gt_distribution, gradesData.maxGrade, 400, 120);
        }} else {{
            gtHistContainer.innerHTML = '<em class="text-gray">No distribution data available</em>';
        }}

        // Populate EGF tabs
        gradesData.egfNames.forEach((egfName, idx) => {{
            const tabContent = document.getElementById('tab-egf-' + idx);
            if (!tabContent) return;

            const gradeDetail = gradesData.egfGrades[egfName]?.[submissionId];
            const gradeValueEl = tabContent.querySelector('.egf-grade-value');
            const histContainer = tabContent.querySelector('.egf-histogram');
            const justificationEl = tabContent.querySelector('.egf-justification');

            if (gradeDetail) {{
                gradeValueEl.textContent = gradeDetail.grade;
                histContainer.innerHTML = generateHistogramSVG(gradeDetail.grade_distribution, gradesData.maxGrade, 400, 120);
                justificationEl.innerHTML = gradeDetail.justification
                    ? parseMarkdown(gradeDetail.justification)
                    : '<em class="text-gray">No justification provided</em>';
            }} else {{
                gradeValueEl.textContent = 'N/A';
                histContainer.innerHTML = '<em class="text-gray">Not graded</em>';
                justificationEl.innerHTML = '<em class="text-gray">No data available</em>';
            }}
        }});

        // Show first tab
        switchTab('essay');

        // Show modal
        document.getElementById('gradeModal').classList.add('active');
        document.body.style.overflow = 'hidden';
    }}

    // Close modal
    function closeModal() {{
        document.getElementById('gradeModal').classList.remove('active');
        document.body.style.overflow = '';
    }}

    // Switch tab
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-btn').forEach(btn => {{
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        }});
        document.querySelectorAll('.tab-content').forEach(content => {{
            content.classList.toggle('active', content.id === 'tab-' + tabId);
        }});
    }}

    // Event listeners
    document.addEventListener('DOMContentLoaded', function() {{
        // Table row clicks
        document.querySelectorAll('.grade-row').forEach(row => {{
            row.addEventListener('click', () => openModal(row.dataset.submissionId));
        }});

        // Tab clicks
        document.querySelectorAll('.tab-btn').forEach(btn => {{
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        }});

        // Close button
        document.querySelector('.modal-close')?.addEventListener('click', closeModal);

        // Close on overlay click
        document.getElementById('gradeModal')?.addEventListener('click', (e) => {{
            if (e.target.classList.contains('modal-overlay')) closeModal();
        }});

        // Close on Escape
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});
    }});

    // Expose functions globally
    window.openModal = openModal;
    window.closeModal = closeModal;
    window.switchTab = switchTab;
}})();
</script>
'''


def generate_grades_table_section(grades_table: GradesTableData) -> str:
    """Generate the HTML for the grades table section."""
    # Build table header
    header_cells = ['<th>Student</th>', '<th>GT</th>']
    for egf_name in grades_table.egf_names:
        label = grades_table.egf_labels.get(egf_name, egf_name)
        header_cells.append(f'<th title="{egf_name}">{label}</th>')

    # Build table rows
    rows = []
    for sid, submission in grades_table.submissions.items():
        student_display = submission.student_name or submission.student_id or sid
        gt_grade = submission.ground_truth_grade
        gt_display = str(gt_grade) if gt_grade is not None else '-'

        cells = [
            f'<td>{student_display}</td>',
            f'<td class="grade-cell">{gt_display}</td>',
        ]

        for egf_name in grades_table.egf_names:
            egf_grade = grades_table.egf_grades.get(egf_name, {}).get(sid)
            if egf_grade:
                grade = egf_grade.grade
                # Color based on difference from ground truth
                css_class = 'grade-cell'
                if gt_grade is not None:
                    diff = abs(grade - gt_grade)
                    if diff == 0:
                        css_class += ' grade-match'
                    elif diff == 1:
                        css_class += ' grade-close'
                    else:
                        css_class += ' grade-far'
                cells.append(f'<td class="{css_class}">{grade}</td>')
            else:
                cells.append('<td class="grade-cell">-</td>')

        rows.append(f'<tr class="grade-row" data-submission-id="{sid}">{"".join(cells)}</tr>')

    return f'''
    <section class="grades-section">
        <h2>Grades by Submission</h2>
        <p style="color: var(--gray-600); font-size: 0.875rem; margin-bottom: 1rem;">
            Click a row to view details. Colors: <span style="color: var(--success);">exact match</span>,
            <span style="color: var(--warning);">within 1</span>,
            <span style="color: var(--danger);">2+ difference</span>
        </p>
        <div class="table-container">
            <table class="grades-table">
                <thead>
                    <tr>{"".join(header_cells)}</tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
    </section>
    '''


def generate_modal_html(grades_table: GradesTableData) -> str:
    """Generate the HTML for the modal popup."""
    # Build tab buttons
    tab_buttons = [
        '<button class="tab-btn active" data-tab="essay">Essay</button>',
        '<button class="tab-btn" data-tab="gt">Ground Truth</button>',
    ]
    for idx, egf_name in enumerate(grades_table.egf_names):
        label = grades_table.egf_labels.get(egf_name, egf_name)
        tab_buttons.append(f'<button class="tab-btn" data-tab="egf-{idx}" title="{egf_name}">{label}</button>')

    # Build tab contents
    egf_tab_contents = []
    for idx, egf_name in enumerate(grades_table.egf_names):
        label = grades_table.egf_labels.get(egf_name, egf_name)
        egf_tab_contents.append(f'''
            <div class="tab-content" id="tab-egf-{idx}">
                <div class="grade-display">
                    <span class="grade-value egf-grade-value">-</span>
                    <span class="grade-max">/ {grades_table.max_grade}</span>
                </div>
                <div class="histogram-section">
                    <h4>Grade Distribution</h4>
                    <div class="histogram-container egf-histogram"></div>
                </div>
                <div class="justification-section">
                    <h4>Justification</h4>
                    <div class="markdown-content egf-justification"></div>
                </div>
            </div>
        ''')

    return f'''
    <div class="modal-overlay" id="gradeModal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Student Details</h3>
                <button class="modal-close" aria-label="Close">&times;</button>
            </div>
            <div class="modal-tabs">
                {"".join(tab_buttons)}
            </div>
            <div class="modal-body">
                <div class="tab-content active" id="tab-essay">
                    <div class="markdown-content" id="essayContent"></div>
                </div>
                <div class="tab-content" id="tab-gt">
                    <div class="grade-display">
                        <span class="grade-value" id="gtGradeValue">-</span>
                        <span class="grade-max" id="gtGradeMax">/ -</span>
                    </div>
                    <div class="histogram-section">
                        <h4>Teacher Noise Distribution (<span id="gtNoiseLabel">expected</span>)</h4>
                        <div class="histogram-container" id="gtHistogram"></div>
                    </div>
                </div>
                {"".join(egf_tab_contents)}
            </div>
        </div>
    </div>
    '''


def save_html_report(result: FullAnalysisResult, output_path: Path, noise_assumption: str = "expected") -> None:
    """Save analysis results as a self-contained HTML file."""
    html = generate_html(result, noise_assumption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
