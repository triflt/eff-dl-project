#!/usr/bin/env python3
"""
Simple script to test a model on Set5 and measure quality + latency.

Usage:
    # Test FP32 model
    python scripts/test_set5.py --checkpoint ./results/ESPCN_x4-T91/g_best.pth.tar --mode fp32
    
    # Test QAT model (fake quant)
    python scripts/test_set5.py --checkpoint ./results/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar --mode qat --method lsq
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


def load_model(checkpoint_path: str, mode: str, method: str = None) -> nn.Module:
    """Load model based on mode."""
    if mode == "fp32":
        prev_qat = getattr(config, "qat_enabled", False)
        try:
            config.qat_enabled = False
            model = model_def.__dict__[config.model_arch_name](
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                channels=config.channels
            )
        finally:
            config.qat_enabled = prev_qat
    
    elif mode == "qat":
        if method is None:
            raise ValueError("--method is required for QAT mode")
        
        prev_qat = getattr(config, "qat_enabled", False)
        prev_method = getattr(config, "qat_method", "lsq")
        try:
            config.qat_enabled = True
            config.qat_method = method
            model = model_def.__dict__[config.model_arch_name](
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                channels=config.channels
            )
        finally:
            config.qat_enabled = prev_qat
            config.qat_method = prev_method
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    return model


def evaluate_set5(model: nn.Module, device: torch.device) -> dict:
    """Evaluate model on Set5."""
    psnr_metric = PSNR(config.upscale_factor, config.only_test_y_channel).to(device)
    ssim_metric = SSIM(config.upscale_factor, config.only_test_y_channel).to(device)
    
    test_ds = TestImageDataset(
        test_gt_images_dir=f"./data/Set5/GTmod12",
        test_lr_images_dir=f"./data/Set5/LRbicx{config.upscale_factor}"
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    model = model.to(device)
    model.eval()
    
    psnr_total, ssim_total = 0.0, 0.0
    
    with torch.inference_mode():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = model(lr)
            
            # Handle NaN/Inf
            sr = torch.nan_to_num(sr, nan=0.0, posinf=1.0, neginf=0.0)
            sr = torch.clamp(sr, 0.0, 1.0)
            
            psnr_total += psnr_metric(sr, gt).item()
            ssim_total += ssim_metric(sr, gt).item()
    
    n = len(test_loader)
    return {
        "psnr": psnr_total / n,
        "ssim": ssim_total / n,
    }


def measure_latency(model: nn.Module, device: torch.device, iterations: int = 100) -> float:
    """Measure inference latency."""
    model = model.to(device)
    model.eval()
    
    h = config.gt_image_size // config.upscale_factor if hasattr(config, "gt_image_size") else 17
    dummy_input = torch.randn(1, config.in_channels, h, h).to(device)
    
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
    parser = argparse.ArgumentParser(description="Test model on Set5")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, required=True, choices=["fp32", "qat"],
                        help="Model mode: fp32 or qat")
    parser.add_argument("--method", type=str, choices=["lsq", "pact", "apot", "efficientqat"],
                        help="QAT method (required for qat mode)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    args = parser.parse_args()
    
    # Validate
    if args.mode == "qat" and args.method is None:
        parser.error("--method is required for qat mode")
    
    # Device selection
    if args.device == "auto":
        device = config.device
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"Testing on Set5")
    print(f"{'='*60}")
    print(f"Mode:       {args.mode.upper()}")
    if args.method:
        print(f"Method:     {args.method.upper()}")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.mode, args.method)
    print("✓ Model loaded\n")
    
    # Evaluate
    print("Evaluating on Set5...")
    metrics = evaluate_set5(model, device)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"{'='*60}\n")
    
    # Measure latency
    print("Measuring latency...")
    latency = measure_latency(model, device, iterations=50)
    print(f"Latency: {latency:.2f} ms/image\n")


if __name__ == "__main__":
    main()

