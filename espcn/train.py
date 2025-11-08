# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time

import torch
from torch import nn
from torch import optim
from contextlib import nullcontext
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, CPUPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    espcn_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = None
    scheduler = None

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        # If QAT is enabled, load FP32 checkpoint into FP32 model and copy into QAT model
        if getattr(config, "qat_enabled", False):
            # Build FP32 model temporarily
            prev_qat = config.qat_enabled
            config.qat_enabled = False
            fp32_model = build_model()
            fp32_model = load_state_dict(fp32_model, config.pretrained_model_weights_path, load_mode="pretrained")
            # Restore QAT flag and rebuild QAT model fresh
            config.qat_enabled = prev_qat
            espcn_model = build_model()
            # Copy conv weights from FP32 into QAT inner convs
            _copy_fp32_to_qat(fp32_model, espcn_model)
            # Recreate optimizer/scheduler bound to the new model
            optimizer = define_optimizer(espcn_model)
            scheduler = define_scheduler(optimizer)
            print(f"Loaded FP32 pretrained weights into QAT model from `{config.pretrained_model_weights_path}`.")
        else:
            espcn_model = load_state_dict(espcn_model, config.pretrained_model_weights_path, load_mode="pretrained")
            optimizer = define_optimizer(espcn_model)
            scheduler = define_scheduler(optimizer)
            print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")
        # Create optimizer/scheduler if not yet created
        if optimizer is None:
            optimizer = define_optimizer(espcn_model)
            print("Define all optimizer functions successfully.")
        if scheduler is None:
            scheduler = define_scheduler(optimizer)
            print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume_model_weights_path:
        espcn_model, _, start_epoch, best_psnr, best_ssim, optimizer, _ = load_state_dict(
            espcn_model,
            config.resume_model_weights_path,
            optimizer=optimizer,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize autocast and gradient scaler according to device
    if config.device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        class DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()
        autocast = nullcontext

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train(espcn_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              autocast)
        psnr, ssim = validate(espcn_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test",
                              autocast)
        print("\n")

        # Update lr
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": espcn_model.state_dict(),
                         "optimizer": optimizer.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)

def _copy_fp32_to_qat(fp32_model: nn.Module, qat_model: nn.Module) -> None:
    """Copy Conv2d weights/bias from FP32 ESPCN into QAT-wrapped ESPCN (inner conv)."""
    # Flatten conv modules in order for both models
    def list_convs(module: nn.Module):
        return [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    def list_qconvs(module: nn.Module):
        # QAConv2d has attribute `conv` which is nn.Conv2d
        qconvs = []
        for m in module.modules():
            if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
                qconvs.append(getattr(m, "conv"))
        return qconvs
    fp32_convs = list_convs(fp32_model)
    qat_inner_convs = list_qconvs(qat_model)
    num = min(len(fp32_convs), len(qat_inner_convs))
    with torch.no_grad():
        for i in range(num):
            qat_inner_convs[i].weight.copy_(fp32_convs[i].weight)
            if fp32_convs[i].bias is not None and qat_inner_convs[i].bias is not None:
                qat_inner_convs[i].bias.copy_(fp32_convs[i].bias)

def load_dataset() -> [CUDAPrefetcher | CPUPrefetcher, CUDAPrefetcher | CPUPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_gt_images_dir,
                                            config.gt_image_size,
                                            config.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(config.test_gt_images_dir, config.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    if config.device.type == "cuda":
        train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    espcn_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                         out_channels=config.out_channels,
                                                         channels=config.channels)
    espcn_model = espcn_model.to(device=config.device)

    return espcn_model


def define_loss() -> nn.MSELoss:
    criterion = nn.MSELoss()
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(espcn_model) -> optim.SGD:
    optimizer = optim.SGD(espcn_model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay,
                          nesterov=config.model_nesterov)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.MultiStepLR:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_scheduler_milestones,
                                         gamma=config.lr_scheduler_gamma)

    return scheduler


def train(
        espcn_model: nn.Module,
        train_prefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler,
        writer: SummaryWriter,
        autocast
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    espcn_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        espcn_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with autocast():
            sr = espcn_model(lr)
            loss = torch.mul(config.loss_weights, criterion(sr, gt))
            # PACT alpha regularization (canonical)
            if getattr(config, "qat_enabled", False) and getattr(config, "qat_method", "") == "pact":
                alpha_reg = float(getattr(config, "pact_alpha_reg", 0.0) or 0.0)
                if alpha_reg > 0.0:
                    reg_term = 0.0
                    # Import locally to avoid hard dependency when not using PACT
                    try:
                        from pact.quant import PACTQuantizer
                        for m in espcn_model.modules():
                            if isinstance(m, PACTQuantizer):
                                reg_term = reg_term + F.softplus(m.alpha).pow(2).sum()
                    except Exception:
                        reg_term = 0.0
                    if isinstance(reg_term, torch.Tensor):
                        loss = loss + alpha_reg * reg_term

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        espcn_model: nn.Module,
        data_prefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str,
        autocast
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    espcn_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with autocast():
                sr = espcn_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()