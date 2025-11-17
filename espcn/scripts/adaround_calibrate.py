import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader

import config
import model as model_def
from dataset import TrainValidImageDataset
from utils import make_directory, save_checkpoint
from adaround.quant import calibrate_adaround


def main():
    device = config.device

    # Build FP32 model and load weights (force-disable QAT wrappers)
    prev_qat = getattr(config, "qat_enabled", False)
    try:
        config.qat_enabled = False
        sr_model = model_def.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                              out_channels=config.out_channels,
                                                              channels=config.channels).to(device)
    finally:
        config.qat_enabled = prev_qat
    assert config.pretrained_model_weights_path, "Set pretrained_model_weights_path in config.py to a FP32 checkpoint."
    checkpoint = torch.load(config.pretrained_model_weights_path, map_location="cpu")
    sr_model.load_state_dict(checkpoint["state_dict"])
    sr_model.eval()

    # Build small calibration loader from train dataset
    calib_dataset = TrainValidImageDataset(config.train_gt_images_dir,
                                           config.gt_image_size,
                                           config.upscale_factor,
                                           "Valid")
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

    # Calibrate AdaRound and bake quantized weights
    baked_model = calibrate_adaround(sr_model, calib_loader, device=device, steps=200, bit=8, lr=1e-2)

    # Save baked checkpoint
    exp = f"{config.exp_name}-AdaRound"
    samples_dir = os.path.join("samples", exp)
    results_dir = os.path.join("results", exp)
    make_directory(samples_dir)
    make_directory(results_dir)
    save_checkpoint({
        "epoch": 0,
        "best_psnr": 0.0,
        "best_ssim": 0.0,
        "state_dict": baked_model.state_dict(),
        "optimizer": {},
    }, "adaround_baked.pth.tar", samples_dir, results_dir, "g_best.pth.tar", "g_last.pth.tar", is_best=True, is_last=True)
    print(f"Saved AdaRound-baked model to `{results_dir}`.")


if __name__ == "__main__":
    main()

