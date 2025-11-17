#!/usr/bin/env python3
"""
Quantize an ONNX model (static or dynamic) using ONNX Runtime tooling.

Example:
    python scripts/quantize_onnx_model.py \
        --onnx ./results/ESPCN_x4-T91-LSQ-QAT/onnx_export/espcn_lsq_fp32.onnx \
        --output ./results/ESPCN_x4-T91-LSQ-QAT/onnx_export/espcn_lsq_int8.onnx \
        --mode static \
        --calib-dir ./data/Set5/LRbicx4 \
        --input-shape 1 1 64 64 \
        --samples 20
"""
from __future__ import annotations

import argparse
import glob
import pathlib
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


class LRImageDataReader(CalibrationDataReader):
    """Feeds low-resolution images into the ONNX model for calibration."""

    def __init__(self, image_paths: Iterable[pathlib.Path], input_name: str, input_shape: Tuple[int, int, int, int]):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.input_shape = input_shape
        self._iter = iter(self.image_paths)

    def get_next(self) -> Dict[str, np.ndarray] | None:
        try:
            path = next(self._iter)
        except StopIteration:
            return None

        _, _, h, w = self.input_shape
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read calibration image: {path}")
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(self.input_shape)
        return {self.input_name: img}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model with ONNX Runtime.")
    parser.add_argument("--onnx", type=str, required=True, help="Input FP32 ONNX model path.")
    parser.add_argument("--output", type=str, required=True, help="Destination INT8 ONNX model path.")
    parser.add_argument("--mode", type=str, default="static", choices=["static", "dynamic"], help="Quantization mode.")
    parser.add_argument("--calib-dir", type=str, default="./data/Set5/LRbicx4", help="Directory with LR images for calibration.")
    parser.add_argument("--input-name", type=str, default="lr", help="Input tensor name.")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 1, 64, 64],
        metavar=("N", "C", "H", "W"),
        help="Input tensor shape (default: 1 1 64 64).",
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of calibration samples to use (static mode).")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weight quantization when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = pathlib.Path(args.onnx).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("ONNX Quantization")
    print("============================================================")
    print(f"Input model : {onnx_path}")
    print(f"Output model: {output_path}")
    print(f"Mode        : {args.mode}")
    print("============================================================")

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=output_path.as_posix(),
            per_channel=args.per_channel,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
        print("✓ Dynamic quantization complete")
        return

    # Static quantization
    calib_dir = pathlib.Path(args.calib_dir).expanduser().resolve()
    image_paths = sorted(
        pathlib.Path(p)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for p in glob.glob(str(calib_dir / ext))
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}")
    image_paths = image_paths[: args.samples]

    data_reader = LRImageDataReader(image_paths, args.input_name, tuple(args.input_shape))
    quantize_static(
        model_input=onnx_path.as_posix(),
        model_output=output_path.as_posix(),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )
    print(f"✓ Static quantization complete using {len(image_paths)} samples")


if __name__ == "__main__":
    main()

