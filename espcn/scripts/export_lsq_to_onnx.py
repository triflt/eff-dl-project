#!/usr/bin/env python3
"""
Export an LSQ-QAT ESPCN checkpoint to ONNX for further INT8 optimization.

Usage example:
    python scripts/export_lsq_to_onnx.py \
        --checkpoint ./results/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar \
        --output ./results/ESPCN_x4-T91-LSQ-QAT/onnx_export/espcn_lsq_fp32.onnx
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import train  # noqa: E402
from utils import load_state_dict, make_directory  # noqa: E402


def _capture_config() -> Dict[str, Any]:
    """Snapshot the config fields that we temporarily mutate."""
    return {
        "mode": config.mode,
        "device": config.device,
        "qat_enabled": config.qat_enabled,
        "qat_method": config.qat_method,
        "qat_bits": config.qat_bits,
        "qat_quantize_activations": getattr(config, "qat_quantize_activations", True),
    }


def _restore_config(snapshot: Dict[str, Any]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value)


def build_lsq_model(device: torch.device) -> torch.nn.Module:
    """Build ESPCN with LSQ fake quant enabled."""
    snapshot = _capture_config()
    try:
        config.mode = "test"
        config.device = device
        config.qat_enabled = True
        config.qat_method = "lsq"
        if not hasattr(config, "qat_bits"):
            config.qat_bits = 8
        if not hasattr(config, "qat_quantize_activations"):
            config.qat_quantize_activations = True
        config.qat_quantize_activations = True

        espcn_model = train.build_model()
        espcn_model.to(device)
        return espcn_model
    finally:
        _restore_config(snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LSQ ESPCN checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the LSQ QAT checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination ONNX file path.",
    )
    parser.add_argument(
        "--lr-size",
        type=int,
        default=64,
        help="Spatial size of the low-resolution dummy input used for tracing (HxW).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = pathlib.Path(args.output).expanduser().resolve()
    make_directory(output_path.parent.as_posix())

    device = torch.device("cpu")
    model = build_lsq_model(device)
    model = load_state_dict(model, checkpoint_path.as_posix(), load_mode="pretrained")
    model.eval()

    lr_size = args.lr_size
    dummy_input = torch.randn(1, config.in_channels, lr_size, lr_size, device=device)

    dynamic_axes = {
        "lr": {0: "batch", 2: "lr_height", 3: "lr_width"},
        "sr": {0: "batch", 2: "sr_height", 3: "sr_width"},
    }

    print("============================================================")
    print("Exporting LSQ ESPCN to ONNX")
    print("============================================================")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output     : {output_path}")
    print(f"Dummy input: (1, {config.in_channels}, {lr_size}, {lr_size})")
    print("============================================================")

    # Initialize LSQ quantizers before tracing to avoid in-graph copy_ ops.
    with torch.no_grad():
        model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["lr"],
        output_names=["sr"],
        dynamic_axes=dynamic_axes,
    )

    print("✓ ONNX export complete")


if __name__ == "__main__":
    main()

