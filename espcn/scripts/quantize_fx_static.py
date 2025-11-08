import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torch.utils.data import DataLoader

import config
import model as model_def
from dataset import TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import make_directory
import cv2


def main():
    torch.backends.quantized.engine = "fbgemm" if torch.backends.mkldnn.is_available() else "qnnpack"
    device = torch.device("cpu")

    # Build FP32 model and load weights (force-disable QAT wrappers if enabled)
    prev_qat = getattr(config, "qat_enabled", False)
    try:
        config.qat_enabled = False
        fp32 = model_def.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                          out_channels=config.out_channels,
                                                          channels=config.channels).to(device)
    finally:
        config.qat_enabled = prev_qat
    assert hasattr(config, "model_weights_path") or hasattr(config, "pretrained_model_weights_path")
    ckpt_path = getattr(config, "model_weights_path", "") or getattr(config, "pretrained_model_weights_path", "")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    fp32.load_state_dict(state, strict=False)
    fp32.eval()

    # PTQ: prepare with observers
    qconfig_mapping = get_default_qconfig_mapping(torch.backends.quantized.engine)
    # Determine example input size: use train gt_image_size when available, else infer from LR dataset
    if hasattr(config, "gt_image_size"):
        h = config.gt_image_size // config.upscale_factor
        w = h
    else:
        # Infer from first LR image in test dir
        lr_files = sorted(os.listdir(config.lr_dir))
        assert len(lr_files) > 0, "No LR images found for calibration shape inference."
        lr_path = os.path.join(config.lr_dir, lr_files[0])
        img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Failed to read {lr_path}"
        h, w = img.shape[:2]
    example_inputs = (torch.randn(1, config.in_channels, h, w),)
    prepared = prepare_fx(fp32, qconfig_mapping, example_inputs=example_inputs)

    # Calibrate on a few samples
    if hasattr(config, "train_gt_images_dir") and hasattr(config, "gt_image_size"):
        calib_dataset = TrainValidImageDataset(config.train_gt_images_dir,
                                               config.gt_image_size,
                                               config.upscale_factor,
                                               "Valid")
        calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    else:
        calib_dataset = TestImageDataset(config.gt_dir, config.lr_dir)
        calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    with torch.inference_mode():
        for i, batch in enumerate(calib_loader):
            lr = batch["lr"].to(device)
            _ = prepared(lr)
            if i >= 200:
                break

    quantized = convert_fx(prepared)

    # Save quantized model
    exp = f"{config.exp_name}-FXINT8"
    out_dir = os.path.join("results", exp)
    make_directory(out_dir)
    torch.save({"state_dict": quantized.state_dict()}, os.path.join(out_dir, "fx_int8.pth.tar"))

    # Quick CPU quality eval on Set5
    psnr_m = PSNR(config.upscale_factor, config.only_test_y_channel).to(device)
    ssim_m = SSIM(config.upscale_factor, config.only_test_y_channel).to(device)
    test_ds = TestImageDataset(f"./data/Set5/GTmod12", f"./data/Set5/LRbicx{config.upscale_factor}")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    quantized.eval()
    psnr_total, ssim_total = 0.0, 0.0
    with torch.inference_mode():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = quantized(lr)
            psnr_total += psnr_m(sr, gt).item()
            ssim_total += ssim_m(sr, gt).item()
    n = len(test_loader)
    print(f"FX INT8 Set5: PSNR {psnr_total / n:.2f} SSIM {ssim_total / n:.4f}")

    # Latency benchmark
    import time
    iters = 50
    # Use same inferred size for latency input
    lr = torch.randn(1, config.in_channels, h, w)
    start = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            _ = quantized(lr)
    t = (time.time() - start) / iters * 1000.0
    print(f"FX INT8 latency (ms/img): {t:.2f}")


if __name__ == "__main__":
    main()

