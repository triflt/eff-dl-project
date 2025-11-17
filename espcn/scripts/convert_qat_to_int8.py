#!/usr/bin/env python3
"""
Simple script to convert QAT models to real INT8 without FX calibration.
Validation only on Set5 for simplicity.

Usage:
    python scripts/convert_qat_to_int8.py --method lsq --checkpoint ./results/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import model as model_def
from dataset import TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import make_directory


def convert_model_to_int8(model_qat: nn.Module, method: str) -> nn.Module:
    """
    Convert QAT model to INT8 by replacing all QAConv2d with Conv2dInt.
    
    Args:
        model_qat: QAT model with QAConv2d modules
        method: "lsq" or "pact"
    
    Returns:
        model_int8: INT8 model with Conv2dInt modules
    """
    if method == "lsq":
        from lsq.quant import QAConv2d, Conv2dInt
    elif method == "pact":
        from pact.quant import QAConv2d, Conv2dInt
    elif method == "apot":
        from apot.quant import QAConv2d, Conv2dInt
    elif method == "efficientqat":
        from efficientqat.quant import QAConv2d, Conv2dInt
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Replace all QAConv2d in model_qat with Conv2dInt
    def replace_qat_conv(module: nn.Module, prefix: str = "") -> nn.Module:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QAConv2d):
                # Convert to INT8
                conv_int = Conv2dInt.from_qat(child, int_dtype=torch.int8)
                setattr(module, name, conv_int)
                print(f"  Converted {full_name}: QAConv2d -> Conv2dInt")
            else:
                # Recursively replace in child modules
                replace_qat_conv(child, full_name)
        return module
    
    # Convert model_qat in-place by replacing QAConv2d with Conv2dInt
    model_int8 = replace_qat_conv(model_qat)
    
    return model_int8


def optimize_model_with_compile(model: nn.Module) -> nn.Module:
    """Apply torch.compile() optimization if available."""
    if hasattr(torch, 'compile'):
        try:
            print("Applying torch.compile() optimization...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  torch.compile() failed: {e}")
    else:
        print("⚠️  torch.compile() not available (requires PyTorch 2.0+)")
    
    return model


def evaluate_model(model: nn.Module, device: torch.device) -> dict:
    """Evaluate model on Set5 and return metrics."""
    psnr_metric = PSNR(config.upscale_factor, config.only_test_y_channel).to(device)
    ssim_metric = SSIM(config.upscale_factor, config.only_test_y_channel).to(device)
    
    test_ds = TestImageDataset(
        test_gt_images_dir=f"./data/Set5/GTmod12",
        test_lr_images_dir=f"./data/Set5/LRbicx{config.upscale_factor}"
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    model.eval()
    psnr_total, ssim_total = 0.0, 0.0
    
    with torch.inference_mode():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = model(lr)
            psnr_total += psnr_metric(sr, gt).item()
            ssim_total += ssim_metric(sr, gt).item()
    
    n = len(test_loader)
    return {
        "psnr": psnr_total / n,
        "ssim": ssim_total / n,
        "num_images": n,
    }


def measure_latency(model: nn.Module, input_size: tuple, device: torch.device, iterations: int = 100) -> float:
    """Measure inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    start = time.time()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    latency_ms = (time.time() - start) / iterations * 1000.0
    return latency_ms


def main():
    parser = argparse.ArgumentParser(description="Convert QAT model to INT8")
    parser.add_argument("--method", type=str, required=True, choices=["lsq", "pact", "apot", "efficientqat"], 
                        help="QAT method used (lsq, pact, apot, or efficientqat)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to QAT checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generate from checkpoint path)")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile() optimization for faster inference")
    args = parser.parse_args()
    
    device = torch.device("cpu")  # INT8 inference is CPU-only
    
    print(f"\n{'='*60}")
    print(f"Converting {args.method.upper()} QAT model to INT8")
    print(f"{'='*60}\n")
    
    # 1) Load QAT model
    print(f"Loading QAT checkpoint: {args.checkpoint}")
    prev_qat = getattr(config, "qat_enabled", False)
    prev_method = getattr(config, "qat_method", "lsq")
    
    try:
        config.qat_enabled = True
        config.qat_method = args.method
        
        model_qat = model_def.__dict__[config.model_arch_name](
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model_qat.load_state_dict(state, strict=False)
        model_qat.eval()
        
        print(f"✓ Loaded QAT model successfully\n")
        
    finally:
        config.qat_enabled = prev_qat
        config.qat_method = prev_method
    
    # 2) Convert to INT8
    print("Converting QAT modules to INT8...")
    model_int8 = convert_model_to_int8(model_qat, args.method)
    model_int8.to(device)
    model_int8.eval()
    print(f"✓ Conversion complete\n")
    
    # Apply torch.compile() optimization if requested
    if args.compile:
        model_int8 = optimize_model_with_compile(model_int8)
        print()
    
    # 3) Evaluate on Set5
    print(f"Evaluating on Set5...")
    metrics = evaluate_model(model_int8, device)
    
    print(f"\n{'='*60}")
    print(f"Results on Set5 (INT8):")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Images: {metrics['num_images']}")
    print(f"{'='*60}\n")
    
    # 4) Measure latency
    h = config.gt_image_size // config.upscale_factor if hasattr(config, "gt_image_size") else 17
    input_size = (config.in_channels, h, h)
    
    print("Measuring latency on CPU...")
    latency = measure_latency(model_int8, input_size, device, iterations=100)
    print(f"  Latency: {latency:.2f} ms/image\n")
    
    # 5) Save INT8 model
    if args.output_dir is None:
        # Auto-generate output directory
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(checkpoint_dir, f"{checkpoint_name}_INT8")
    
    make_directory(args.output_dir)
    output_path = os.path.join(args.output_dir, "model_int8.pth.tar")
    
    torch.save({
        "state_dict": model_int8.state_dict(),
        "method": args.method,
        "metrics": metrics,
        "latency_ms": latency,
        "config": {
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "channels": config.channels,
            "upscale_factor": config.upscale_factor,
        }
    }, output_path)
    
    print(f"✓ Saved INT8 model to: {output_path}")
    
    # 6) Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Method:      {args.method.upper()}")
    print(f"PSNR:        {metrics['psnr']:.2f} dB (Set5)")
    print(f"SSIM:        {metrics['ssim']:.4f}")
    print(f"Latency:     {latency:.2f} ms/image (CPU)")
    print(f"Output:      {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

