"""Watch mode for continuous EGF analysis."""

import sys
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .core import (
    EGFData,
    load_egf_data,
    load_edf_teacher_noise,
    analyze_egf,
    get_default_teacher_noise,
    find_matching_edf,
)
from .edf_cache import EDFCache
from .bootstrap import get_default_stability_vector, bootstrap_qwk_paired
from .html_output import generate_html, FullAnalysisResult


class EGFWatcher(FileSystemEventHandler):
    """Watch for new EGF files and analyze them against a base file."""

    def __init__(
        self,
        base_egf: EGFData,
        edf_cache: EDFCache,
        noise_assumption: str,
        output_callback: Callable[[str, float], None],
    ):
        self.base_egf = base_egf
        self.edf_cache = edf_cache
        self.noise_assumption = noise_assumption
        self.output_callback = output_callback

        edf_path = find_matching_edf(base_egf, edf_cache)
        if edf_path:
            edf_data = load_edf_teacher_noise(edf_path)
            self.teacher_noise = edf_data.teacher_noise.get(
                noise_assumption,
                get_default_teacher_noise()[noise_assumption]
            )
            self.edf_name = edf_data.name
        else:
            self.teacher_noise = get_default_teacher_noise()[noise_assumption]
            self.edf_name = None

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

        print(f"Analyzing {path.name}...")
        p_win = self._compute_p_win(new_egf)
        output_path = self._generate_output(new_egf, p_win)

        self.output_callback(str(output_path), p_win)

    def _compute_p_win(self, new_egf: EGFData) -> float:
        """Compute P(new_egf > base_egf)."""
        base_grades = {g['submission_id']: g for g in self.base_egf.grades if g['ground_truth'] is not None}
        new_grades = {g['submission_id']: g for g in new_egf.grades if g['ground_truth'] is not None}

        common_ids = sorted(set(base_grades.keys()) & set(new_grades.keys()))
        if not common_ids:
            return 0.5

        by_idx = {
            0: (
                [base_grades[sid]['predicted'] for sid in common_ids],
                [base_grades[sid]['ground_truth'] for sid in common_ids],
                common_ids,
            ),
            1: (
                [new_grades[sid]['predicted'] for sid in common_ids],
                [new_grades[sid]['ground_truth'] for sid in common_ids],
                common_ids,
            ),
        }

        stability_vectors = {0: self.stability_vector, 1: self.stability_vector}

        win_matrix, _ = bootstrap_qwk_paired(
            by_idx, stability_vectors, self.teacher_noise, n_iterations=1000
        )

        return win_matrix.get((1, 0), 0.5)

    def _generate_output(self, new_egf: EGFData, p_win: float) -> Path:
        """Generate HTML output for the comparison."""
        from .core import analyze_egf, FullAnalysisResult, ComparisonResult

        base_result = analyze_egf(
            self.base_egf, self.teacher_noise, self.stability_vector,
            self.edf_name, self.noise_assumption
        )
        new_result = analyze_egf(
            new_egf, self.teacher_noise, self.stability_vector,
            self.edf_name, self.noise_assumption
        )

        comparison = ComparisonResult(
            egf_names=[self.base_egf.name, new_egf.name],
            win_matrix={(0, 1): 1 - p_win, (1, 0): p_win, (0, 0): 0.5, (1, 1): 0.5},
            avg_qwk={0: base_result.qwk_result.raw_qwk, 1: new_result.qwk_result.raw_qwk},
            n_iterations=1000,
            n_common_essays=len([g for g in self.base_egf.grades if g['ground_truth'] is not None]),
        )

        full_result = FullAnalysisResult(
            individual_results=[base_result, new_result],
            comparison=comparison,
            labels=['Base', 'New'],
            legend={'Base': self.base_egf.name, 'New': new_egf.name},
        )

        html = generate_html(full_result, self.noise_assumption)
        output_path = new_egf.path.parent / f"{new_egf.name}_vs_base.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path


def run_watch_mode(
    base_egf_path: Path,
    edf_directory: Path,
    noise_assumption: str = "expected",
) -> None:
    """Run watch mode, analyzing new EGF files against a base file."""
    print(f"Loading base EGF: {base_egf_path}")
    base_egf = load_egf_data(base_egf_path)

    print(f"Scanning EDF directory: {edf_directory}")
    edf_cache = EDFCache(edf_directory)

    edf_path = find_matching_edf(base_egf, edf_cache)
    if edf_path:
        print(f"Found matching EDF: {edf_path.name}")
    else:
        print("Warning: No matching EDF found, using default teacher noise")

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

    handler = EGFWatcher(base_egf, edf_cache, noise_assumption, output_callback)
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
