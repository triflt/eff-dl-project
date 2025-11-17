#!/usr/bin/env python3
"""
Benchmark different optimization techniques on Mac to speed up INT8 inference.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import model as model_def
from lsq.quant import Conv2dInt, QAConv2d


def benchmark_model(model: nn.Module, name: str, device: str = "cpu", iterations: int = 200) -> float:
    """Benchmark model latency."""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(20):
            _ = model(dummy_input)
    
    # Benchmark
    if device == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device == "mps":
                torch.mps.synchronize()
    
    latency_ms = (time.time() - start) / iterations * 1000.0
    print(f"{name:30s} {latency_ms:6.2f} ms/image")
    return latency_ms


def convert_qat_to_int8(model_qat):
    """Convert QAT model to INT8."""
    def replace_qat_conv(module: nn.Module, prefix: str = "") -> nn.Module:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QAConv2d):
                conv_int = Conv2dInt.from_qat(child, int_dtype=torch.int8)
                setattr(module, name, conv_int)
            else:
                replace_qat_conv(child, full_name)
        return module
    
    model_int8 = replace_qat_conv(model_qat)
    return model_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Benchmarking Optimization Techniques on Mac")
    print("=" * 70)
    print()
    
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
    
    print("✓ Loaded QAT model\n")
    
    results = {}
    
    # 1. Original QAT on CPU
    print("1️⃣  Testing QAT (fake quant) on CPU...")
    results['QAT CPU'] = benchmark_model(model_qat.cpu(), "QAT CPU", "cpu")
    print()
    
    # 2. Original QAT on MPS
    if torch.backends.mps.is_available():
        print("2️⃣  Testing QAT (fake quant) on MPS...")
        model_qat_mps = model_def.espcn_x4(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels
        )
        model_qat_mps.load_state_dict(checkpoint["state_dict"])
        model_qat_mps.eval()
        results['QAT MPS'] = benchmark_model(model_qat_mps, "QAT MPS", "mps")
        print()
    
    # 3. INT8 on CPU (baseline)
    print("3️⃣  Testing INT8 on CPU...")
    model_int8 = convert_qat_to_int8(model_qat)
    model_int8.eval()
    results['INT8 CPU'] = benchmark_model(model_int8.cpu(), "INT8 CPU", "cpu")
    print()
    
    # 4. INT8 with torch.compile()
    if hasattr(torch, 'compile'):
        print("4️⃣  Testing INT8 with torch.compile()...")
        try:
            model_int8_compiled = torch.compile(model_int8.cpu(), mode='reduce-overhead')
            results['INT8 CPU (compiled)'] = benchmark_model(
                model_int8_compiled, "INT8 CPU (compiled)", "cpu"
            )
        except Exception as e:
            print(f"⚠️  torch.compile() failed: {e}")
        print()
    
    # 5. INT8 with channels_last memory format
    print("5️⃣  Testing INT8 with channels_last...")
    model_int8_cl = convert_qat_to_int8(model_qat)
    model_int8_cl = model_int8_cl.to(memory_format=torch.channels_last)
    model_int8_cl.eval()
    results['INT8 CPU (channels_last)'] = benchmark_model(
        model_int8_cl.cpu(), "INT8 CPU (channels_last)", "cpu"
    )
    print()
    
    # 6. FP32 baseline on CPU
    print("6️⃣  Testing FP32 baseline on CPU...")
    config.qat_enabled = False
    model_fp32 = model_def.espcn_x4(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels
    )
    model_fp32.load_state_dict(checkpoint["state_dict"], strict=False)
    model_fp32.eval()
    results['FP32 CPU'] = benchmark_model(model_fp32.cpu(), "FP32 CPU", "cpu")
    print()
    
    # 7. FP32 with torch.compile()
    if hasattr(torch, 'compile'):
        print("7️⃣  Testing FP32 with torch.compile()...")
        try:
            model_fp32_compiled = torch.compile(model_fp32.cpu(), mode='reduce-overhead')
            results['FP32 CPU (compiled)'] = benchmark_model(
                model_fp32_compiled, "FP32 CPU (compiled)", "cpu"
            )
        except Exception as e:
            print(f"⚠️  torch.compile() failed: {e}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best INT8
    int8_results = {k: v for k, v in results.items() if 'INT8' in k}
    best_int8 = min(int8_results, key=int8_results.get)
    best_int8_latency = int8_results[best_int8]
    
    # FP32 baseline
    fp32_baseline = results.get('FP32 CPU', results.get('FP32 CPU (compiled)', 0))
    
    print(f"\n{'Method':<30s} {'Latency (ms)':<15s} {'vs FP32':<10s}")
    print("-" * 70)
    
    for name, latency in sorted(results.items(), key=lambda x: x[1]):
        speedup = fp32_baseline / latency if fp32_baseline > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"{name:<30s} {latency:>6.2f} ms       {speedup_str:>8s}")
    
    print("\n" + "=" * 70)
    print(f"🏆 Best INT8: {best_int8} ({best_int8_latency:.2f} ms)")
    
    if fp32_baseline > 0:
        speedup = fp32_baseline / best_int8_latency
        if speedup > 1.0:
            print(f"✅ INT8 is {speedup:.2f}x FASTER than FP32!")
        else:
            print(f"⚠️  INT8 is {1/speedup:.2f}x SLOWER than FP32")
            print("   This is normal on Mac without VNNI support")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

