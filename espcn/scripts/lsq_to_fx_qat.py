import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping

import config
import model as model_def
from lsq.quant import LSQQuantizer, QAConv2d, QuantAct
from dataset import TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import make_directory


def list_lsq_convs_and_acts(model_lsq: nn.Module):
    conv_scales = []
    act_scales = []
    for m in model_lsq.modules():
        if isinstance(m, QAConv2d):
            # weight scale (per-tensor)
            conv_scales.append(float(m.quant_w.s.detach().cpu().item()))
        elif isinstance(m, QuantAct):
            act_scales.append(float(m.quant.s.detach().cpu().item()))
    return conv_scales, act_scales


def copy_conv_weights_from_lsq_to_fp32(model_lsq: nn.Module, model_fp32: nn.Module):
    # Gather inner convs from LSQ model and plain convs from FP32 in order
    lsq_convs = []
    for m in model_lsq.modules():
        if isinstance(m, QAConv2d):
            lsq_convs.append(m.conv)
    fp32_convs = [m for m in model_fp32.modules() if isinstance(m, nn.Conv2d)]
    n = min(len(lsq_convs), len(fp32_convs))
    with torch.no_grad():
        for i in range(n):
            fp32_convs[i].weight.copy_(lsq_convs[i].weight)
            if fp32_convs[i].bias is not None and lsq_convs[i].bias is not None:
                fp32_convs[i].bias.copy_(lsq_convs[i].bias)


def init_fx_observers_from_lsq(prepared: nn.Module, conv_scales, act_scales):
    # Map LSQ weight scales to prepared conv.weight_fake_quant observers
    qmax_w = 127.0  # int8 symmetric
    conv_idx = 0
    for name, mod in prepared.named_modules():
        # weight fake quant is an attribute on conv modules in prepared QAT
        if isinstance(mod, nn.Conv2d) and hasattr(mod, "weight_fake_quant"):
            if conv_idx < len(conv_scales):
                s = conv_scales[conv_idx]
                ap = mod.weight_fake_quant.activation_post_process
                with torch.no_grad():
                    mn_value = -s * qmax_w
                    mx_value = s * qmax_w
                    # support both per-tensor and per-channel observers
                    if hasattr(ap, "min_vals") and hasattr(ap, "max_vals"):
                        # per-channel: match shape
                        ap.min_vals.copy_(torch.full_like(ap.min_vals, mn_value))
                        ap.max_vals.copy_(torch.full_like(ap.max_vals, mx_value))
                    elif hasattr(ap, "min_val") and hasattr(ap, "max_val"):
                        # per-tensor: expand scalar to proper shape
                        ap.min_val.copy_(torch.full_like(ap.min_val, mn_value))
                        ap.max_val.copy_(torch.full_like(ap.max_val, mx_value))
                conv_idx += 1
    # Optionally map activation scales (best-effort)
    # FX inserts activation fake_quant modules; we set their observers from LSQ act scales in order.
    qmax_a = 127.0
    act_idx = 0
    for name, mod in prepared.named_modules():
        # Activation fake quant modules are instances of FakeQuantize
        if mod.__class__.__name__.endswith("FakeQuantize"):
            if act_idx < len(act_scales):
                s = act_scales[act_idx]
                ap = getattr(mod, "activation_post_process", None)
                if ap is not None:
                    with torch.no_grad():
                        mn_value = -s * qmax_a
                        mx_value = s * qmax_a
                        if hasattr(ap, "min_vals") and hasattr(ap, "max_vals"):
                            ap.min_vals.copy_(torch.full_like(ap.min_vals, mn_value))
                            ap.max_vals.copy_(torch.full_like(ap.max_vals, mx_value))
                        elif hasattr(ap, "min_val") and hasattr(ap, "max_val"):
                            ap.min_val.copy_(torch.full_like(ap.min_val, mn_value))
                            ap.max_val.copy_(torch.full_like(ap.max_val, mx_value))
                act_idx += 1


def main():
    torch.backends.quantized.engine = "fbgemm" if torch.backends.mkldnn.is_available() else "qnnpack"
    device = torch.device("cpu")

    # 1) Build LSQ model and load LSQ-QAT checkpoint
    prev_qat = getattr(config, "qat_enabled", False)
    prev_method = getattr(config, "qat_method", "lsq")
    try:
        config.qat_enabled = True
        config.qat_method = "lsq"
        model_lsq = model_def.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                               out_channels=config.out_channels,
                                                               channels=config.channels).to(device)
    finally:
        config.qat_enabled = prev_qat
        config.qat_method = prev_method
    lsq_ckpt = getattr(config, "pretrained_model_weights_path", "")
    assert lsq_ckpt, "Set pretrained_model_weights_path to your LSQ-QAT checkpoint in config.py"
    state = torch.load(lsq_ckpt, map_location="cpu")["state_dict"]
    model_lsq.load_state_dict(state, strict=False)
    model_lsq.eval()

    # Extract LSQ scales and conv weights
    conv_scales, act_scales = list_lsq_convs_and_acts(model_lsq)

    # 2) Build plain FP32 model and copy conv weights
    config.qat_enabled = False
    model_fp32 = model_def.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                            out_channels=config.out_channels,
                                                            channels=config.channels).to(device)
    copy_conv_weights_from_lsq_to_fp32(model_lsq, model_fp32)
    model_fp32.eval()

    # 3) Prepare QAT FX and initialize observers from LSQ
    qconfig_mapping = get_default_qat_qconfig_mapping(torch.backends.quantized.engine)
    h = config.gt_image_size // config.upscale_factor
    example_inputs = (torch.randn(1, config.in_channels, h, h),)
    prepared = prepare_qat_fx(model_fp32, qconfig_mapping, example_inputs=example_inputs)
    init_fx_observers_from_lsq(prepared, conv_scales, act_scales)

    # 4) Short FX-QAT finetune
    train_ds = TrainValidImageDataset(config.train_gt_images_dir,
                                      config.gt_image_size,
                                      config.upscale_factor,
                                      "Train")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(prepared.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4, nesterov=False)
    prepared.train()
    epochs = 3
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = prepared(lr)
            loss = criterion(sr, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} Iter {i+1}: loss {running / 100:.6f}")
                running = 0.0

    # 5) Convert to INT8 and save
    int8_model = convert_fx(prepared.eval())
    exp = f"{config.exp_name}-LSQ2FXQATINT8"
    out_dir = os.path.join("results", exp)
    make_directory(out_dir)
    torch.save({"state_dict": int8_model.state_dict()}, os.path.join(out_dir, "lsq_fx_qat_int8.pth.tar"))
    print(f"Saved LSQ→FX-QAT INT8 model to {out_dir}")

    # 6) Evaluate on Set5 (CPU)
    psnr_m = PSNR(config.upscale_factor, config.only_test_y_channel).to(device)
    ssim_m = SSIM(config.upscale_factor, config.only_test_y_channel).to(device)
    test_ds = TestImageDataset(f"./data/Set5/GTmod12", f"./data/Set5/LRbicx{config.upscale_factor}")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    psnr_total, ssim_total = 0.0, 0.0
    int8_model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            gt = batch["gt"].to(device)
            sr = int8_model(lr)
            psnr_total += psnr_m(sr, gt).item()
            ssim_total += ssim_m(sr, gt).item()
    n = len(test_loader)
    print(f"LSQ→FX-QAT INT8 Set5: PSNR {psnr_total / n:.2f} SSIM {ssim_total / n:.4f}")


if __name__ == "__main__":
    main()

