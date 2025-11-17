#!/usr/bin/env python3
"""
Benchmark ONNX / ONNX Runtime inference latency.

Usage example:
    python scripts/benchmark_onnx_runtime.py \
        --onnx ./results/ESPCN_x4-T91-LSQ-QAT/onnx_export/espcn_lsq_fp32.onnx \
        --device cpu --runs 100 --warmup 20 --input-shape 1 1 64 64
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Sequence

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime latency.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution provider to use. (Note: CUDA provider requires GPU build of ONNX Runtime.)",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of timed runs.")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up iterations.")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        default=[1, 1, 64, 64],
        help="Input tensor shape used for benchmarking.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of intra/inter-op threads to use (set 0 to leave default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible dummy inputs.",
    )
    return parser.parse_args()


def select_providers(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def benchmark(
    session: ort.InferenceSession,
    input_shape: Sequence[int],
    warmup: int,
    runs: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    def sample_input() -> np.ndarray:
        return rng.standard_normal(input_shape, dtype=np.float32)

    # Warm-up
    for _ in range(warmup):
        session.run(output_names, {input_name: sample_input()})

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(runs):
        data = sample_input()
        start = time.perf_counter()
        session.run(output_names, {input_name: data})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
    return float(np.mean(latencies_ms))


def main() -> None:
    args = parse_args()
    providers = select_providers(args.device)

    print("============================================================")
    print("ONNX Runtime Latency Benchmark")
    print("============================================================")
    print(f"Model      : {args.onnx}")
    print(f"Providers  : {providers}")
    print(f"Input shape: {tuple(args.input_shape)}")
    print(f"Warmup     : {args.warmup}")
    print(f"Runs       : {args.runs}")
    print(f"Threads    : {args.threads if args.threads > 0 else 'auto'}")
    print("============================================================")

    os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))

    sess_options = ort.SessionOptions()
    if args.threads > 0:
        sess_options.intra_op_num_threads = args.threads
        sess_options.inter_op_num_threads = args.threads
    sess_options.log_severity_level = 2

    session = ort.InferenceSession(args.onnx, providers=providers, sess_options=sess_options)
    avg_latency = benchmark(session, args.input_shape, args.warmup, args.runs, args.seed)

    print(f"✓ Average latency: {avg_latency:.3f} ms over {args.runs} runs")


if __name__ == "__main__":
    main()

