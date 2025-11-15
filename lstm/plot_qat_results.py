#!/usr/bin/env python3
"""Generate comparison plots for QAT experiments stored under lstm/artifacts."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt

QAT_METHODS = ["lsq", "pact", "adaround", "apot", "efficientqat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot QAT losses, metrics, and timing comparisons."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default="artifacts",
        help="Directory that contains per-run artifact folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Where to store the generated figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=QAT_METHODS,
        help="QAT methods to include (defaults to lsq, pact, adaround, apot, efficientqat).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def iter_method_runs(artifacts_dir: Path, method: str) -> List[Path]:
    prefix = f"{method}_"
    return sorted(
        path
        for path in artifacts_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


def plot_loss_curves(method: str, run_dirs: Sequence[Path], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False

    for run_dir in run_dirs:
        json_path = run_dir / "qat_loss.json"
        if not json_path.exists():
            continue
        loss_data = load_json(json_path)
        y_values = loss_data.get("y")
        if not y_values:
            continue
        x_values = loss_data.get("x")
        if not x_values or len(x_values) != len(y_values):
            x_values = list(range(1, len(y_values) + 1))
        plt.plot(x_values, y_values, label=run_dir.name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(f"{method.upper()} QAT Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_base_metrics(artifacts_dir: Path) -> float:
    """Return the final recorded metric value from the baseline run."""
    base_metrics: Dict[str, float] = {}
    metrics_path = artifacts_dir / "base" / "metrics.json"
    if not metrics_path.exists():
        return base_metrics

    raw_metrics = load_json(metrics_path)
    for name, values in raw_metrics.items():
        if isinstance(values, list):
            if not values:
                continue
            base_metrics[name] = float(values[-1])
        elif isinstance(values, (int, float)):
            base_metrics[name] = float(values)
    return base_metrics['train_acc']


def plot_metric_curves(
    method: str,
    run_dirs: Sequence[Path],
    output_path: Path,
    base_metrics: float,
) -> None:
    metric_series: Dict[str, List[tuple[str, List[int], List[float]]]] = OrderedDict()

    for run_dir in run_dirs:
        json_path = run_dir / "metrics.json"
        if not json_path.exists():
            continue
        metrics = load_json(json_path)
        for metric_name, values in metrics.items():
            if not isinstance(values, list):
                continue
            xs = list(range(len(values)))
            ys = [float(v) for v in values]
            metric_series.setdefault(metric_name, []).append((run_dir.name, xs, ys))

    if not metric_series:
        return

    metric_names = list(metric_series.keys())
    fig, axes = plt.subplots(
        len(metric_names), 1, figsize=(10, 4 * len(metric_names)), squeeze=False
    )

    for ax, metric_name in zip(axes.flatten(), metric_names):
        for label, xs, ys in sorted(metric_series[metric_name], key=lambda item: item[0]):
            ax.scatter(xs, ys, label=label)
        ax.scatter([0], [base_metrics], label="base", marker="x", color="black")
        ax.set_title(f"{method.upper()} {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def collect_timing_runs(
    artifacts_dir: Path, methods: Iterable[str]
) -> List[tuple[str, Dict[str, float]]]:
    run_names = ["base"] + [f"{method}_1_1e-3" for method in methods]
    normalized: List[tuple[str, Dict[str, float]]] = []

    for run_name in run_names:
        run_dir = artifacts_dir / run_name
        json_path = run_dir / "timings.json"
        if not json_path.exists():
            continue
        timings = load_json(json_path)
        label = "base" if run_name == "base" else run_name.split("_", 1)[0]
        entry: Dict[str, float] = {}
        if run_name == "base":
            if "train" in timings:
                entry["train"] = timings["train"]
            if "infer" in timings:
                entry["infer"] = timings["infer"]
        else:
            if "qat_train" in timings:
                entry["train"] = timings["qat_train"]
            if "qat_infer" in timings:
                entry["infer"] = timings["qat_infer"]
            if "quantized_infer" in timings:
                entry["quantized_infer"] = timings["quantized_infer"]
        normalized.append((label, entry))
    return normalized


def plot_timing_bars(
    timing_runs: Sequence[tuple[str, Dict[str, float]]],
    metric: str,
    output_path: Path,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for label, entries in timing_runs:
        value = entries.get(metric)
        if value is None:
            continue
        labels.append(label)
        values.append(value)

    if not values:
        return

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4a90e2")
    pretty_name = metric.replace("_", " ").title()
    plt.title(f"{pretty_name} Timing")
    plt.ylabel("Seconds")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_metrics = load_base_metrics(artifacts_dir)

    for method in args.methods:
        run_dirs = iter_method_runs(artifacts_dir, method)
        if not run_dirs:
            print(f"[WARN] No runs found for method '{method}'.")
            continue
        method_out = output_dir / method
        plot_loss_curves(method, run_dirs, method_out / f"{method}_losses.png")
        plot_metric_curves(
            method, run_dirs, method_out / f"{method}_metrics.png", base_metrics
        )

    timing_runs = collect_timing_runs(artifacts_dir, args.methods)
    timing_out = output_dir / "timings"
    for metric in ("train", "infer", "quantized_infer"):
        plot_timing_bars(timing_runs, metric, timing_out / f"{metric}.png")


if __name__ == "__main__":
    main()
