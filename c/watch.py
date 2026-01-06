"""Watch mode for continuous EGF analysis."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .core import (
    EGFData,
    Essay,
    load_egf_data,
    load_edf_teacher_noise,
    load_edf_ground_truth,
    build_essays,
    analyze_essays,
    get_default_teacher_noise,
    find_matching_edf,
    load_edf_submissions_detail,
    load_egf_grades_detail,
    build_grades_table_data,
)
from .edf_cache import EDFCache
from .bootstrap import get_default_stability_vector, bootstrap_qwk_paired
from .html_output import generate_html, FullAnalysisResult
from .main import generate_summary_markdown


class EGFWatcher(FileSystemEventHandler):
    """Watch for new EGF files and analyze them against a base file."""

    def __init__(
        self,
        base_egf: EGFData,
        edf_path: Path,
        ground_truth: dict[str, int],
        noise_assumption: str,
        output_callback: Callable[[str, float], None],
    ):
        self.base_egf = base_egf
        self.base_essays = build_essays(base_egf, ground_truth)
        self.edf_path = edf_path
        self.ground_truth = ground_truth
        self.noise_assumption = noise_assumption
        self.output_callback = output_callback

        edf_data = load_edf_teacher_noise(self.edf_path)
        self.teacher_noise = edf_data.teacher_noise.get(
            noise_assumption,
            get_default_teacher_noise()[noise_assumption]
        )
        self.edf_name = edf_data.name

        self.stability_vector = get_default_stability_vector()

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle new file creation."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".egf":
            return

        time.sleep(0.5)

        try:
            new_egf = load_egf_data(path)
        except Exception as e:
            print(f"Warning: Failed to load {path.name}: {e}", file=sys.stderr)
            return

        if new_egf.source_hash != self.base_egf.source_hash:
            print(f"Skipping {path.name}: different source EDF", file=sys.stderr)
            return

        # Build essays by combining predicted grades (EGF) with ground truth (EDF)
        new_essays = build_essays(new_egf, self.ground_truth)

        print(f"Analyzing {path.name}...")
        p_win = self._compute_p_win(new_essays)
        output_path = self._generate_output(new_egf, new_essays, p_win)

        self.output_callback(str(output_path), p_win)

    def _compute_p_win(self, new_essays: list[Essay]) -> float:
        """Compute P(new > base)."""
        base_grades = {e.submission_id: e for e in self.base_essays}
        new_grades = {e.submission_id: e for e in new_essays}

        common_ids = sorted(set(base_grades.keys()) & set(new_grades.keys()))
        if not common_ids:
            return 0.5

        by_idx = {
            0: (
                [base_grades[sid].predicted for sid in common_ids],
                [base_grades[sid].ground_truth for sid in common_ids],
                common_ids,
            ),
            1: (
                [new_grades[sid].predicted for sid in common_ids],
                [new_grades[sid].ground_truth for sid in common_ids],
                common_ids,
            ),
        }

        stability_vectors = {0: self.stability_vector, 1: self.stability_vector}

        win_matrix, _ = bootstrap_qwk_paired(
            by_idx, stability_vectors, self.teacher_noise, n_iterations=1000
        )

        return win_matrix.get((1, 0), 0.5)

    def _generate_output(self, new_egf: EGFData, new_essays: list[Essay], p_win: float) -> Path:
        """Generate HTML output for the comparison."""
        from .core import FullAnalysisResult, ComparisonResult

        base_result = analyze_essays(
            self.base_essays, self.base_egf, self.teacher_noise, self.stability_vector,
            self.edf_name, self.noise_assumption
        )
        new_result = analyze_essays(
            new_essays, new_egf, self.teacher_noise, self.stability_vector,
            self.edf_name, self.noise_assumption
        )

        comparison = ComparisonResult(
            egf_names=[self.base_egf.name, new_egf.name],
            win_matrix={(0, 1): 1 - p_win, (1, 0): p_win, (0, 0): 0.5, (1, 1): 0.5},
            avg_qwk={0: base_result.qwk_result.raw_qwk, 1: new_result.qwk_result.raw_qwk},
            n_iterations=1000,
            n_common_essays=len(self.base_essays),
        )

        full_result = FullAnalysisResult(
            individual_results=[base_result, new_result],
            comparison=comparison,
            labels=['Base', 'New'],
            legend={'Base': self.base_egf.name, 'New': new_egf.name},
        )

        # Build grades table data
        if self.edf_path:
            try:
                edf_submissions = load_edf_submissions_detail(self.edf_path, self.noise_assumption)
                egf_grades_list = [
                    (self.base_egf.name, load_egf_grades_detail(self.base_egf.path)),
                    (new_egf.name, load_egf_grades_detail(new_egf.path)),
                ]
                max_grade = self.base_egf.max_grade
                full_result.grades_table = build_grades_table_data(
                    edf_submissions,
                    egf_grades_list,
                    full_result.labels,
                    max_grade,
                    self.noise_assumption,
                )
            except Exception as e:
                print(f"Warning: Failed to build grades table: {e}", file=sys.stderr)

        # Generate summary markdown
        full_result.summary_markdown = generate_summary_markdown(full_result)

        html = generate_html(full_result, self.noise_assumption)
        output_path = new_egf.path.parent / f"{new_egf.name}_vs_base.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path


def run_watch_mode(
    base_egf_path: Path,
    edf_directory: Path,
    noise_assumption: str = "expected",
    quiet: bool = False,
) -> None:
    """Run watch mode, analyzing new EGF files against a base file."""
    print(f"Loading base EGF: {base_egf_path}")
    base_egf = load_egf_data(base_egf_path)

    print(f"Scanning EDF directory: {edf_directory}")
    edf_cache = EDFCache(edf_directory)

    edf_path = find_matching_edf(base_egf, edf_cache)
    if not edf_path:
        print("Error: No matching EDF found. EDF is required for ground truth.", file=sys.stderr)
        sys.exit(1)

    print(f"Found matching EDF: {edf_path.name}")
    print(f"Loading ground truth from: {edf_path.name}")
    ground_truth = load_edf_ground_truth(edf_path)

    watch_dir = edf_directory

    def output_callback(output_path: str, p_win: float) -> None:
        print(f"\n{'='*60}")
        print(f"Output: {output_path}")
        print(f"P(New > Base) = {p_win:.1%}")
        if p_win > 0.5:
            print("Result: New file appears BETTER than base")
        elif p_win < 0.5:
            print("Result: New file appears WORSE than base")
        else:
            print("Result: New file appears SIMILAR to base")
        print(f"{'='*60}\n")

        # Open in browser
        if not quiet:
            subprocess.run(['open', str(Path(output_path).resolve())])

    handler = EGFWatcher(base_egf, edf_path, ground_truth, noise_assumption, output_callback)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)

    print(f"\nWatching {watch_dir} for new EGF files...")
    print("Press Ctrl+C to stop\n")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching.")

    observer.join()
