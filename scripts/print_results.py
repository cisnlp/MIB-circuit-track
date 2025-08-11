"""
Score-aggregation utilities for the MIB Circuit Interp-Bench experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from tabulate import tabulate

from MIB_circuit_track.utils import COL_MAPPING, load_pickle

ResultDict = Dict[str, Any]
Score = Union[ResultDict, str]

_HEADER = [
    "Method",
    "IOI (GPT)",
    "IOI (QWen)",
    "IOI (Gemma)",
    "IOI (Llama)",
    "MCQA (QWen)",
    "MCQA (Gemma)",
    "MCQA (Llama)",
    "Arithmetic (Llama)",
    "ARC-E (Gemma)",
    "ARC-E (Llama)",
    "ARC-C (Llama)",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv (Sequence[str] | None): Custom argument list for testing.
            Uses ``sys.argv`` if *None*.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate CPR / CMD / AUROC scores into human-readable tables."
    )

    # Dataset config
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Which dataset split to read.",
    )
    # Metrics config
    parser.add_argument(
        "--metric",
        choices=["cpr", "cmd", "auroc"],
        default="cpr",
        help=(
            "Which metric to aggregate: "
            "'cpr' for Cumulative Precision-Recall, "
            "'cmd' for Cumulative Mass-Drop, "
            "or 'auroc' for Interp-Bench AUROC."
        ),
    )

    # Output config
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing sub-folders for each method.",
    )

    return parser.parse_args(argv)


def area_under_curve(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute the normalised area under a discrete curve using the trapezoidal rule.

    Args:
        x (Sequence[float]): Monotonically increasing x-coordinates.
        y (Sequence[float]): y-coordinates corresponding to *x*.

    Returns:
        float: Area under *y(x)*, normalised to ``[0, 1]``.

    Raises:
        ValueError: If *x* and *y* differ in length or contain fewer than two points.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("Need at least two points to integrate a curve.")

    xmax = x[-1]
    area = 0.0
    for i in range(len(x) - 1):
        x1, x2 = x[i] / xmax, x[i + 1] / xmax
        area += (x2 - x1) * (y[i] + y[i + 1]) / 2
    return area


def discover_methods(output_dir: Path) -> List[Path]:
    """
    List every method directory contained in *output_dir*.

    Args:
        output_dir (Path): Root directory created by your experiment runner.

    Returns:
        List[Path]: Paths of immediate sub-directories.
    """
    return [p for p in output_dir.iterdir() if p.is_dir()]


def collect_interpbench_scores(
    output_dir: Path,
    split: str,
) -> Dict[str, Union[float, str]]:
    """
    Collect AUROC scores for **Interp-Bench** runs only.

    Args:
        output_dir (Path): Root directory produced by the experiments.
        split (str): Dataset split, e.g. ``"validation"``.

    Returns:
        Dict[str, Union[float, str]]:
            Mapping ``method_name → AUROC`` or ``"-"`` if unavailable.
    """
    scores: Dict[str, Union[float, str]] = {}

    for method_dir in discover_methods(output_dir):
        method_name = method_dir.name
        scores.setdefault(method_name, "-")

        for results_file in method_dir.glob("*_interpbench_*_*.pkl"):
            task, model, file_split, abs_tag = results_file.stem.split("_")
            if file_split != split:
                continue

            results = load_pickle(results_file)
            auroc = area_under_curve(results["FPR"], results["TPR"])
            scores[method_name] = auroc

    return scores


def build_interpbench_table(scores: Dict[str, Union[float, str]]) -> str:
    """
    Build a tabulated string for Interp-Bench AUROC scores.

    Args:
        scores (Dict[str, Union[float, str]]): ``method → score`` mapping.

    Returns:
        str: Printable table.
    """
    header = ["Method", "IOI (InterpBench)"]
    rows = [[m, f"{s:.2f}" if isinstance(s, float) else s] for m, s in scores.items()]
    return tabulate([header, *rows])


def _should_consider_file(
    task: str,
    model: str,
    split: str,
    desired_split: str,
) -> bool:
    """
    Decide whether a result file should be processed.

    Args:
        task (str): Task name, e.g. ``"ioi"``.
        model (str): Model name, e.g. ``"gpt"`` or ``"interpbench"``.
        split (str): Split encoded in the filename.
        desired_split (str): Split specified by the CLI.

    Returns:
        bool: ``True`` if the file matches the CLI filters.
    """
    key = f"{task}_{model}"
    return key in COL_MAPPING and model != "interpbench" and split == desired_split


def _keep_based_on_metric(
    metric: str,
    abs_flag: str,
    file_path: Path,
) -> bool:
    """
    Determine whether to **retain** this result file for aggregation.

    The experiment runner writes one file with ``abs-True`` and one with
    ``abs-False``.  For *CPR* we want the *relative* scores, whereas for
    *CMD* we want the *absolute* ones.

    Args:
        metric (str): Either ``"cpr"`` or ``"cmd"``.
        abs_flag (str): ``"True"`` or ``"False"`` extracted from the filename.
        file_path (Path): Full path to the results file.

    Returns:
        bool: ``True`` if the file should be loaded.
    """
    if metric == "cpr" and abs_flag == "True":
        # A relative CPR score exists iff the complementary file (abs-False)
        # does not exist.
        return not file_path.with_name(
            file_path.name.replace("abs-True", "abs-False")
        ).exists()

    if metric == "cmd" and abs_flag == "False":
        # A CMD score exists iff the complementary file (abs-True)
        # does not exist.
        return not file_path.with_name(
            file_path.name.replace("abs-False", "abs-True")
        ).exists()

    return metric == "cmd" if abs_flag == "True" else metric == "cpr"


def collect_task_scores(
    output_dir: Path, split: str, metric: str, n_cols: int = 12
) -> Dict[str, List[Score]]:
    """
    Aggregate scores for all tasks *except* Interp-Bench.

    Args:
        output_dir (Path): Root directory of experiment outputs.
        split (str): Dataset split filter.
        metric (str): Either ``"cpr"`` or ``"cmd"``.

    Returns:
        Dict[str, List[Score]]:
            Mapping ``method_name → list`` where the list index is
            ``COL_MAPPING[col_key]`` and each element is either a result
            dict or the placeholder ``"-"``.
    """
    scores: Dict[str, List[Score]] = {}

    for method_dir in discover_methods(output_dir):
        method_name = method_dir.name
        scores.setdefault(method_name, ["-"] * n_cols)

        for file_path in method_dir.glob("*.pkl"):
            parts = file_path.stem.split("_")
            if len(parts) != 4:
                continue  # Skip malformed filenames

            task, model, file_split, abs_tag = parts
            if not _should_consider_file(task, model, file_split, split):
                continue

            abs_flag = abs_tag.split("abs-")[1]
            if not _keep_based_on_metric(metric, abs_flag, file_path):
                continue

            results = load_pickle(file_path)
            col_idx = COL_MAPPING[f"{task}_{model}"]

            if col_idx is not None:
                scores[method_name][col_idx] = results

    return scores


def build_task_table(
    scores: Dict[str, List[Score]],
    metric: str,
) -> str:
    """
    Build a tabulated string for *CPR* or *CMD* scores.

    Args:
        scores (Dict[str, List[Score]]): Output of :func:`collect_task_scores`.
        metric (str): Either ``"cpr"`` or ``"cmd"``.

    Returns:
        str: Printable table.
    """
    rows: List[List[str]] = []

    for method_name, row_scores in scores.items():
        if metric == "cpr":
            formatted = [
                f"{s['area_under']:.2f}" if isinstance(s, dict) else "-"
                for s in row_scores
            ]
        else:  # CMD
            formatted = [
                f"{s['area_from_1']:.2f}" if isinstance(s, dict) else "-"
                for s in row_scores
            ]
        rows.append([method_name, *formatted])

    return tabulate([_HEADER, *rows])


def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry-point.

    Args:
        argv (Sequence[str] | None): Optional custom ``sys.argv`` replacement.
    """
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.metric == "auroc":
        interpbench_scores = collect_interpbench_scores(output_dir, args.split)
        print(build_interpbench_table(interpbench_scores))
    else:
        # otherwise aggregate CPR / CMD
        scores = collect_task_scores(output_dir, args.split, args.metric)
        print(build_task_table(scores, args.metric))


if __name__ == "__main__":
    main()
