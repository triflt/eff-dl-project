#!/usr/bin/env python3
"""Generate comparison plots for QAT experiments stored under lstm/artifacts."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    parser.add_argument(
        "--comparison-metric",
        default="quantized_acc",
        help="Metric key to use when comparing the best run for each QAT method.",
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
            xs = list(range(1, len(values) + 1))
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
        ax.scatter([1], [base_metrics], label="base", marker="x", color="black")
        ax.set_title(f"{method.upper()} {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def collect_best_metrics(
    artifacts_dir: Path, methods: Iterable[str], metric_name: str
) -> "OrderedDict[str, Tuple[float, str]]":
    """Return the best metric per QAT method."""
    best_metrics: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
    for method in methods:
        run_dirs = iter_method_runs(artifacts_dir, method)
        best_value: Optional[float] = None
        best_run: Optional[str] = None
        for run_dir in run_dirs:
            json_path = run_dir / "metrics.json"
            if not json_path.exists():
                continue
            metrics = load_json(json_path)
            values = metrics.get(metric_name)
            if values is None:
                continue
            if isinstance(values, list):
                if not values:
                    continue
                candidate = max(float(v) for v in values)
            elif isinstance(values, (int, float)):
                candidate = float(values)
            else:
                continue
            if best_value is None or candidate > best_value:
                best_value = candidate
                best_run = run_dir.name
        if best_value is not None and best_run is not None:
            best_metrics[method] = (best_value, best_run)
    return best_metrics


def plot_best_metric_bars(
    best_metrics: "OrderedDict[str, Tuple[float, str]]",
    metric_name: str,
    output_path: Path,
) -> None:
    if not best_metrics:
        return
    methods = list(best_metrics.keys())
    values = [best_metrics[method][0] for method in methods]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(methods, values, color="#7f8c8d")
    pretty_metric = metric_name.replace("_", " ").title()
    plt.title(f"Best {pretty_metric} by QAT Method")
    plt.ylabel(pretty_metric)
    plt.ylim(0.95, 0.99)
    for bar, method in zip(bars, methods):
        value, run_name = best_metrics[method]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}\n{run_name}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


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

    best_metrics = collect_best_metrics(
        artifacts_dir, args.methods, args.comparison_metric
    )
    if not best_metrics:
        print(
            f"[WARN] Could not find metric '{args.comparison_metric}' for the provided methods."
        )
    else:
        plot_best_metric_bars(
            best_metrics,
            args.comparison_metric,
            output_dir / f"best_{args.comparison_metric}.png",
        )

    timing_runs = collect_timing_runs(artifacts_dir, args.methods)
    timing_out = output_dir / "timings"
    for metric in ("train", "infer", "quantized_infer"):
        plot_timing_bars(timing_runs, metric, timing_out / f"{metric}.png")


if __name__ == "__main__":
    main()
