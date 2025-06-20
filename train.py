import random

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.nn.functional import softmax
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
)

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from data import ImageFolder
from model import SuperCool, Bouncer
from loss import TVLoss

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Generative adversarial training script.")

    parser.add_argument(
        "--base_model_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=2, type=int)
    parser.add_argument("--target_resolution", default=512, type=int)
    parser.add_argument("--brightness_jitter", default=0.1, type=float)
    parser.add_argument("--contrast_jitter", default=0.1, type=float)
    parser.add_argument("--saturation_jitter", default=0.1, type=float)
    parser.add_argument("--hue_jitter", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--critic_warmup_epochs", default=3, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--task_sampling_temperature", default=1.0, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--num_channels", default=128, type=int)
    parser.add_argument("--hidden_ratio", default=2, choices={1, 2, 4}, type=int)
    parser.add_argument("--num_encoder_layers", default=20, type=int)
    parser.add_argument(
        "--critic_model_size", default="small", choices={"small", "medium", "large"}
    )
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/fine-tuned.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    dtype = (
        torch.bfloat16
        if args.device == "cuda" and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    pre_transformer = Compose(
        [
            RandomResizedCrop(args.target_resolution),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=args.brightness_jitter,
                contrast=args.contrast_jitter,
                saturation=args.saturation_jitter,
                hue=args.hue_jitter,
            ),
        ]
    )

    training = ImageFolder(
        root_path=args.train_images_path,
        upscale_ratio=args.upscale_ratio,
        target_resolution=args.target_resolution,
        pre_transformer=pre_transformer,
    )
    testing = ImageFolder(
        root_path=args.test_images_path,
        upscale_ratio=args.upscale_ratio,
        target_resolution=args.target_resolution,
    )

    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=True,
        num_workers=args.num_dataset_processes,
    )
    test_loader = DataLoader(
        testing,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=False,
        num_workers=args.num_dataset_processes,
    )

    upscaler_args = {
        "upscale_ratio": args.upscale_ratio,
        "num_channels": args.num_channels,
        "hidden_ratio": args.hidden_ratio,
        "num_encoder_layers": args.num_encoder_layers,
    }

    upscaler = SuperCool(**upscaler_args)

    print("Compiling upscaler model")
    upscaler = torch.compile(upscaler)

    critic_args = {
        "model_size": args.critic_model_size,
    }

    critic = Bouncer(**critic_args)

    print("Compiling critic model")
    critic = torch.compile(critic)

    upscaler = upscaler.to(args.device)
    critic = critic.to(args.device)

    l2_loss_function = MSELoss()
    bce_loss_function = BCEWithLogitsLoss()
    tv_loss_function = TVLoss()

    upscaler_optimizer = AdamW(upscaler.parameters(), lr=args.learning_rate)
    critic_optimizer = AdamW(critic.parameters(), lr=args.learning_rate)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=True
        )  # Always load into CPU RAM first to prevent CUDA out-of-memory errors.

        upscaler.load_state_dict(checkpoint["upscaler"])
        upscaler_optimizer.load_state_dict(checkpoint["upscaler_optimizer"])

        critic.load_state_dict(checkpoint["critic"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        starting_epoch += checkpoint["epoch"]

        upscaler = upscaler.to(args.device)
        critic = critic.to(args.device)

        print("Previous checkpoint resumed successfully")

    print(f"Upscaler has {upscaler.num_trainable_params:,} trainable parameters")
    print(f"Critic has {critic.num_trainable_params:,} trainable parameters")

    psnr_metric = PeakSignalNoiseRatio().to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    print("Training ...")
    upscaler.train()
    critic.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_l2_loss, total_tv_loss = 0.0, 0.0
        total_u_bce_loss, total_c_bce_loss = 0.0, 0.0
        total_u_gradient_norm, total_c_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            real_labels = torch.full((x.size(0), 1), 1.0).to(args.device)
            fake_labels = torch.full((x.size(0), 1), 0.0).to(args.device)

            with amp_context:
                c_pred_real = critic.forward(y)

                u_pred = upscaler.forward(x)

                c_pred_fake = critic.forward(u_pred.detach())

                c_pred = torch.cat((c_pred_real, c_pred_fake), dim=0)
                labels = torch.cat((real_labels, fake_labels), dim=0)

                c_bce_loss = bce_loss_function(c_pred, labels)

                c_loss = c_bce_loss

                scaled_c_loss = c_loss / args.gradient_accumulation_steps

            scaled_c_loss.backward()

            update_this_step = step % args.gradient_accumulation_steps == 0

            if update_this_step:
                norm = clip_grad_norm_(critic.parameters(), args.max_gradient_norm)

                critic_optimizer.step()

                critic_optimizer.zero_grad()

                total_c_gradient_norm += norm.item()
                total_steps += 1

            total_c_bce_loss += c_bce_loss.item()

            if epoch > args.critic_warmup_epochs:
                with amp_context:
                    l2_loss = l2_loss_function(u_pred, y)

                    c_pred = critic.forward(u_pred)

                    u_bce_loss = bce_loss_function(c_pred, real_labels)

                    tv_loss = tv_loss_function(u_pred)

                    normalized_losses = torch.stack(
                        [
                            l2_loss / l2_loss.detach(),
                            u_bce_loss / u_bce_loss.detach(),
                            tv_loss / tv_loss.detach(),
                        ]
                    )

                    r = torch.randn(3, device=args.device)

                    r /= args.task_sampling_temperature

                    task_weights = softmax(r, dim=0)

                    weighted_losses = task_weights * normalized_losses

                    u_loss = weighted_losses.sum()

                    scaled_u_loss = u_loss / args.gradient_accumulation_steps

                scaled_u_loss.backward()

                if update_this_step:
                    norm = clip_grad_norm_(
                        upscaler.parameters(), args.max_gradient_norm
                    )

                    upscaler_optimizer.step()

                    total_u_gradient_norm += norm.item()

                total_l2_loss += l2_loss.item()
                total_u_bce_loss += u_bce_loss.item()
                total_tv_loss += tv_loss.item()

            if update_this_step:
                upscaler_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)

            total_batches += 1

        average_l2_loss = total_l2_loss / total_batches
        average_u_bce_loss = total_u_bce_loss / total_batches
        average_c_bce_loss = total_c_bce_loss / total_batches
        average_tv_loss = total_tv_loss / total_batches

        average_u_gradient_norm = total_u_gradient_norm / total_steps
        average_c_gradient_norm = total_c_gradient_norm / total_steps

        logger.add_scalar("Reconstruction L2", average_l2_loss, epoch)
        logger.add_scalar("Upscaler BCE", average_u_bce_loss, epoch)
        logger.add_scalar("Critic BCE", average_c_bce_loss, epoch)
        logger.add_scalar("TV Loss", average_tv_loss, epoch)
        logger.add_scalar("Upscaler Norm", average_u_gradient_norm, epoch)
        logger.add_scalar("Critic Norm", average_c_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Reconstruction L2: {average_l2_loss:.5},",
            f"Upscaler BCE: {average_u_bce_loss:.5},",
            f"Critic BCE: {average_c_bce_loss:.5},",
            f"TV Loss: {average_tv_loss:.5},",
            f"Upscaler Norm: {average_u_gradient_norm:.4},",
            f"Critic Norm: {average_c_gradient_norm:.4}",
        )

        if epoch % args.eval_interval == 0:
            upscaler.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    y_pred = upscaler(x)

                    psnr_metric.update(y_pred, y)
                    ssim_metric.update(y_pred, y)
                    vif_metric.update(y_pred, y)

            psnr = psnr_metric.compute()
            ssim = ssim_metric.compute()
            vif = vif_metric.compute()

            logger.add_scalar("PSNR", psnr, epoch)
            logger.add_scalar("SSIM", ssim, epoch)
            logger.add_scalar("VIF", vif, epoch)

            print(f"PSNR: {psnr:.5}, SSIM: {ssim:.5}, VIF: {vif:.5}")

            upscaler.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "upscaler_args": upscaler_args,
                "upscaler": upscaler.state_dict(),
                "upscaler_optimizer": upscaler_optimizer.state_dict(),
                "critic_args": critic_args,
                "critic": critic.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
