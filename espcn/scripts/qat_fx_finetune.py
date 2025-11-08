import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping

import config
import model as model_def
from dataset import TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import make_directory


def main():
    # QAT FX is CPU-oriented; use CPU for portability and INT8 backend parity
    torch.backends.quantized.engine = "fbgemm" if torch.backends.mkldnn.is_available() else "qnnpack"
    device = torch.device("cpu")

    # 1) Build plain FP32 model (disable custom QAT wrappers) and load FP32 weights
    prev_qat = getattr(config, "qat_enabled", False)
    try:
        config.qat_enabled = False
        model_fp32 = model_def.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                                out_channels=config.out_channels,
                                                                channels=config.channels).to(device)
    finally:
        config.qat_enabled = prev_qat

    assert hasattr(config, "model_weights_path") or hasattr(config, "pretrained_model_weights_path"), \
        "Provide FP32 checkpoint in config.model_weights_path or config.pretrained_model_weights_path"
    ckpt_path = getattr(config, "model_weights_path", "") or getattr(config, "pretrained_model_weights_path", "")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model_fp32.load_state_dict(state, strict=False)
    print(f"Loaded FP32 weights with missing: {missing}, unexpected: {unexpected}")
    model_fp32.eval()

    # 2) Prepare QAT FX
    qconfig_mapping = get_default_qat_qconfig_mapping(torch.backends.quantized.engine)
    example_h = getattr(config, "gt_image_size", 17 * config.upscale_factor) // config.upscale_factor
    example_inputs = (torch.randn(1, config.in_channels, example_h, example_h),)
    model_qat = prepare_qat_fx(model_fp32, qconfig_mapping, example_inputs=example_inputs)

    # 3) QAT finetune (short)
    # Minimal train loader (reuse train dataset in "Valid" mode to avoid random crop)
    train_ds = TrainValidImageDataset(config.train_gt_images_dir,
                                      getattr(config, "gt_image_size", 17 * config.upscale_factor),
                                      config.upscale_factor,
                                      "Train")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model_qat.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=False)

    model_qat.train()
    epochs = 5
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = model_qat(lr)
            loss = criterion(sr, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} Iter {i+1}: loss {running / 50:.6f}")
                running = 0.0

    # 4) Convert to real INT8
    model_int8 = convert_fx(model_qat.eval())

    # 5) Save INT8
    exp = f"{config.exp_name}-FXQATINT8"
    out_dir = os.path.join("results", exp)
    make_directory(out_dir)
    torch.save({"state_dict": model_int8.state_dict()}, os.path.join(out_dir, "fx_qat_int8.pth.tar"))
    print(f"Saved INT8 QAT model to {out_dir}")

    # 6) Evaluate PSNR/SSIM on Set5 and latency on CPU
    psnr_m = PSNR(config.upscale_factor, config.only_test_y_channel).to(device)
    ssim_m = SSIM(config.upscale_factor, config.only_test_y_channel).to(device)
    test_ds = TestImageDataset(f"./data/Set5/GTmod12", f"./data/Set5/LRbicx{config.upscale_factor}")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    psnr_total, ssim_total = 0.0, 0.0
    model_int8.eval()
    with torch.inference_mode():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = model_int8(lr)
            psnr_total += psnr_m(sr, gt).item()
            ssim_total += ssim_m(sr, gt).item()
    n = len(test_loader)
    print(f"FX QAT INT8 Set5: PSNR {psnr_total / n:.2f} SSIM {ssim_total / n:.4f}")

    # Latency
    iters = 50
    lr = torch.randn(1, config.in_channels, example_h, example_h)
    start = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model_int8(lr)
    t = (time.time() - start) / iters * 1000.0
    print(f"FX QAT INT8 latency (ms/img): {t:.2f}")


if __name__ == "__main__":
    main()

