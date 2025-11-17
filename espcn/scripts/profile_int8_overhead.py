#!/usr/bin/env python3
"""
Profile INT8 vs FP32 to understand WHERE the overhead comes from.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import model as model_def
from lsq.quant import Conv2dInt, QAConv2d


def convert_qat_to_int8(model_qat):
    """Convert QAT model to INT8."""
    def replace_qat_conv(module: nn.Module, prefix: str = "") -> nn.Module:
        for name, child in list(module.named_children()):
            if isinstance(child, QAConv2d):
                conv_int = Conv2dInt.from_qat(child, int_dtype=torch.int8)
                setattr(module, name, conv_int)
            else:
                replace_qat_conv(child, f"{prefix}.{name}" if prefix else name)
        return module
    
    return replace_qat_conv(model_qat)


def profile_model(model: nn.Module, name: str, iterations: int = 50):
    """Profile model with PyTorch profiler."""
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256)
    
    print(f"\n{'='*70}")
    print(f"Profiling: {name}")
    print(f"{'='*70}\n")
    
    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(f"model_inference_{name}"):
            with torch.inference_mode():
                for _ in range(iterations):
                    _ = model(dummy_input)
    
    # Print top operations by CPU time
    print(f"\n🔥 Top operations by CPU time:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15,
        top_level_events_only=False
    ))
    
    # Breakdown by operation type
    print(f"\n📊 Breakdown by operation type:")
    events = prof.key_averages()
    
    # Group by operation name
    op_times = {}
    for evt in events:
        op_name = evt.key.split('(')[0].strip()  # Get base op name
        if evt.cpu_time_total > 0:
            if op_name not in op_times:
                op_times[op_name] = 0
            op_times[op_name] += evt.cpu_time_total
    
    # Sort and print
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)[:10]
    total_time = sum(op_times.values())
    
    print(f"{'Operation':<40s} {'Time (ms)':<12s} {'% of Total':<12s}")
    print("-" * 70)
    for op, t in sorted_ops:
        time_ms = t / 1000.0  # Convert to ms
        pct = (t / total_time) * 100 if total_time > 0 else 0
        print(f"{op:<40s} {time_ms:>10.2f} ms {pct:>10.1f}%")


def analyze_single_layer(model_fp32, model_int8):
    """Compare single Conv2d layer: FP32 vs INT8."""
    print(f"\n{'='*70}")
    print("Single Layer Analysis: Conv2d (64, 1, 5x5)")
    print(f"{'='*70}\n")
    
    # Extract first conv layers
    conv_fp32 = model_fp32.feature_maps[0]  # Regular Conv2d
    conv_int8 = model_int8.feature_maps[0]  # Conv2dInt
    
    # Input
    x = torch.randn(1, 1, 256, 256)
    
    # Time FP32 conv
    iterations = 500
    start = time.time()
    for _ in range(iterations):
        with torch.inference_mode():
            _ = conv_fp32(x)
    fp32_time = (time.time() - start) / iterations * 1000
    
    # Time INT8 conv
    start = time.time()
    for _ in range(iterations):
        with torch.inference_mode():
            _ = conv_int8(x)
    int8_time = (time.time() - start) / iterations * 1000
    
    print(f"FP32 Conv2d:  {fp32_time:.3f} ms")
    print(f"INT8 Conv2d:  {int8_time:.3f} ms")
    print(f"Overhead:     {int8_time - fp32_time:.3f} ms ({(int8_time/fp32_time - 1)*100:.1f}%)")
    
    # Analyze INT8 operations
    print(f"\n📊 INT8 Conv2d breakdown:")
    
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            for _ in range(100):
                _ = conv_int8(x)
    
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    print("=" * 70)
    print("INT8 vs FP32 Performance Analysis")
    print("=" * 70)
    
    # Load QAT model
    config.qat_enabled = True
    config.qat_method = "lsq"
    
    model_qat = model_def.espcn_x4(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_qat.load_state_dict(checkpoint["state_dict"])
    model_qat.eval()
    
    # Create FP32 model
    config.qat_enabled = False
    model_fp32 = model_def.espcn_x4(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels
    )
    model_fp32.load_state_dict(checkpoint["state_dict"], strict=False)
    model_fp32.eval()
    
    # Create INT8 model
    model_int8 = convert_qat_to_int8(model_qat)
    model_int8.eval()
    
    print("\n✓ Loaded models")
    
    # 1. Profile full models
    profile_model(model_fp32, "FP32 Baseline", iterations=50)
    profile_model(model_int8, "INT8 (with overhead)", iterations=50)
    
    # 2. Analyze single layer
    analyze_single_layer(model_fp32, model_int8)
    
    # 3. Summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}\n")
    
    print("🔍 Key findings:")
    print()
    print("1. INT8 Conv2d operations:")
    print("   - Dequantize weights: INT8 → FP32")
    print("   - Dequantize inputs: INT8 → FP32  (if quantizer exists)")
    print("   - Standard FP32 convolution")
    print("   - NO native INT8 convolution")
    print()
    print("2. Overhead sources:")
    print("   - Type conversions (INT8 ↔ FP32)")
    print("   - Scale multiplications")
    print("   - Memory allocation for intermediate tensors")
    print()
    print("3. Why FP32 is faster:")
    print("   - Direct FP32 operations (no conversions)")
    print("   - Hardware FP32 units are well-optimized")
    print("   - No INT8 support in PyTorch Conv2d")
    print()
    print("4. Where INT8 WOULD be faster:")
    print("   - Intel CPU with VNNI (DL Boost)")
    print("   - Using ONNX Runtime with INT8 kernels")
    print("   - Using TensorRT on NVIDIA GPU")
    print("   - Mobile processors with INT8 acceleration")
    print()
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

