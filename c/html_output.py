"""Self-contained HTML output generation for analysis results."""

import base64
import io
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from .core import FullAnalysisResult, AnalysisResult, ComparisonResult, QWKResult


def generate_html(result: FullAnalysisResult, noise_assumption: str = "expected") -> str:
    """Generate a self-contained HTML report from analysis results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    html = f'''<!DOCTYPE html>
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

        {"".join(individual_sections)}

        <footer>
            Generated by c - EGF Analysis CLI
        </footer>
    </div>
</body>
</html>'''

    return html


def generate_individual_section(result: AnalysisResult, label: str) -> str:
    """Generate HTML section for an individual EGF analysis."""
    qwk = result.qwk_result

    gt_mean, gt_lower, gt_upper = qwk.gt_noise_ci
    grading_mean, grading_lower, grading_upper = qwk.grading_noise_ci
    sampling_mean, sampling_lower, sampling_upper = qwk.sampling_ci

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
            </tbody>
        </table>
    </section>
    '''


def generate_qwk_bar_chart(result: FullAnalysisResult) -> str:
    """Generate SVG bar chart showing QWK with 3 CI bars per file."""
    n_files = len(result.individual_results)
    if n_files == 0:
        return ""

    bar_width = 20
    group_width = bar_width * 3 + 30
    chart_width = max(400, n_files * group_width + 100)
    chart_height = 300
    margin_left = 50
    margin_bottom = 60
    margin_top = 30

    plot_height = chart_height - margin_top - margin_bottom

    colors = {
        'gt': '#ef4444',
        'grading': '#3b82f6',
        'sampling': '#22c55e',
    }

    bars_svg = []

    for idx, res in enumerate(result.individual_results):
        label = result.labels[idx] if idx < len(result.labels) else str(idx)
        qwk = res.qwk_result

        group_x = margin_left + idx * group_width + 20

        raw_y = margin_top + plot_height * (1 - qwk.raw_qwk)

        bars_svg.append(f'''
            <line x1="{group_x - 5}" y1="{raw_y}" x2="{group_x + bar_width * 3 + 5}" y2="{raw_y}"
                  stroke="#111" stroke-width="2" stroke-dasharray="4"/>
        ''')

        ci_data = [
            ('gt', qwk.gt_noise_ci),
            ('grading', qwk.grading_noise_ci),
            ('sampling', qwk.sampling_ci),
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
            <text x="{group_x + bar_width * 1.5}" y="{chart_height - margin_bottom + 20}"
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
            <line x1="0" y1="50" x2="20" y2="50" stroke="#111" stroke-width="2" stroke-dasharray="4"/>
            <text x="25" y="54" font-size="10">Raw QWK</text>
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


def save_html_report(result: FullAnalysisResult, output_path: Path, noise_assumption: str = "expected") -> None:
    """Save analysis results as a self-contained HTML file."""
    html = generate_html(result, noise_assumption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
