"""AFRO observation pretraining adapted from KolaKivy/AFRO."""
# Commit: 283024a49e2c14bc030debf15383a4e7ee73836d
# Source: afro_workspace/policy/afro.py and AFRO_Pretrain.py.

import signal
import time

import hydra
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet
from dexmani_policy.common.config import positive_int, read_split_manifest
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, set_seed
from dexmani_policy.training.common.data import build_train_loader, close_data_loader, epoch_position
from dexmani_policy.training.common.logging import ExperimentLogger, to_log_scalars
from dexmani_policy.training.common.optimization import build_scheduler
from dexmani_policy.training.pretrain.obs.afro_dit import DiTFeaturePredictor
from dexmani_policy.training.pretrain.obs.common import evaluate, load_run, restore_run, save_run


def vicreg_loss(prediction, target):
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        invariance = F.mse_loss(prediction, target)
        centered = [value - value.mean(0, keepdim=True) for value in (prediction, target)]
        std = [(value.var(0, unbiased=False) + 1e-4).sqrt() for value in centered]
        variance = 0.5 * (F.relu(1 - std[0]).mean() + F.relu(1 - std[1]).mean())
        covariance = prediction.new_zeros(())
        for value in centered:
            n, dim = value.shape
            if n > 1:
                cov = value.T @ value / (n - 1.0)
                covariance = covariance + (cov.square().sum() - cov.diagonal().square().sum()) / dim
        return 25 * invariance + 25 * variance + covariance, invariance, variance, covariance


class AFROPretrain(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.visual = PointNet(3, 64, pooling="max")
        self.teacher = PointNet(3, 64, pooling="max")
        self.teacher.requires_grad_(False)
        self.idm = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 16))
        self.fdm = DiTFeaturePredictor(64)
        self.normalizer = LinearNormalizer(
            {"point_cloud": (3,)} if normalization["point_cloud"] != "identity" else {}
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        )

    def forward(self, batch):
        points = batch["obs"]["point_cloud"]
        if "point_cloud" in self.normalizer.fields:
            points = self.normalizer["point_cloud"].normalize(points)
        batch_size, frames = points.shape[:2]
        points = points.flatten(0, 1)
        online = self.visual(points)["global_token"].reshape(batch_size, frames, 64)
        with torch.no_grad():
            teacher = self.teacher(points)["global_token"].reshape(batch_size, frames, 64)
        losses, metrics = [], {}
        for name, current, following, target in (
            ("forward", online[:, :-1], online[:, 1:], teacher[:, 1:]),
            ("reverse", online[:, 1:], online[:, :-1], teacher[:, :-1]),
        ):
            latent = self.idm(following - current).reshape(-1, 16)
            target = target.reshape(-1, 64)
            timestep = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (len(target),), device=target.device,
            )
            noise = torch.randn_like(target)
            prediction = self.fdm(self.noise_scheduler.add_noise(target, noise, timestep),
                                  timestep, current.reshape(-1, 64), latent)
            loss, inv, var, cov = vicreg_loss(prediction, target)
            losses.append(loss)
            metrics.update({f"{name}_{key}": value.detach()
                            for key, value in zip(("invariance", "variance", "covariance"), (inv, var, cov))})
        loss = losses[0] + losses[1]
        return loss, {"loss": loss.detach(), **metrics}

    @torch.no_grad()
    def update_teacher(self, epoch):
        momentum = 0.996 + 0.004 * min(epoch, 300) / 300
        for target, online in zip(self.teacher.parameters(), self.visual.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)


def train(cfg, checkpoint=None, max_updates=None):
    if cfg.training.loop.gradient_accumulation_steps != 1:
        raise ValueError("AFRO VICReg requires accumulation=1; a smaller batch is an explicit recipe change")
    positive_int(cfg.training.save_every_epochs, "training.save_every_epochs")
    positive_int(cfg.training.val_every_epochs, "training.val_every_epochs")
    set_seed(cfg.training.seed)
    dataset = hydra.utils.instantiate(
        cfg.dataset, frame_stride=cfg.model.frame_stride, _convert_="all",
        split_manifest=read_split_manifest(cfg.dataset.split_manifest),
    )
    cfg.dataset.split_manifest = dataset.split_manifest
    loader = build_train_loader(cfg, dataset)
    batches_per_epoch = loader.batch_sampler.batches_per_epoch
    total = cfg.training.num_epochs * batches_per_epoch
    cfg.training.loop.total_train_steps = total
    model = AFROPretrain(cfg.normalization)
    if checkpoint is None:
        model.normalizer.load_state_dict(dataset.get_normalizer(cfg.normalization).state_dict(), strict=True)
    validation = dataset.get_validation_dataset()
    model.to(cfg.training.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), **cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps, total)
    step = 0
    if checkpoint is not None:
        step = restore_run(model, optimizer, scheduler, checkpoint, device=cfg.training.device)
    epoch, start_batch = epoch_position(step, batches_per_epoch, 1)
    if checkpoint is not None:
        print(f"Resume epoch={epoch}, batch={start_batch}, global_step={step}")
    stop_step = total if max_updates is None else min(total, step + max_updates)
    logger = ExperimentLogger(**cfg.workspace)
    print(f"[AFRO] samples={len(dataset)}, batches_per_epoch={batches_per_epoch}, "
          f"batch={cfg.dataloader.batch_size}, total_updates={total}")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        logger.save_config(cfg)
        while step < stop_step and not stop_requested:
            epoch, start_batch = epoch_position(step, batches_per_epoch, 1)
            loader.batch_sampler.set_epoch(epoch, start_batch)
            model.train()
            current_epoch = epoch
            step_start = time.perf_counter()
            for index, batch in enumerate(loader, start=start_batch):
                data_wait = time.perf_counter() - step_start
                batch = dict_apply(batch, lambda x: x.to(cfg.training.device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = model(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite AFRO loss at step {step}")
                loss.backward()
                gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(gradients).all():
                    raise RuntimeError(f"Nonfinite AFRO gradient at step {step}")
                model.update_teacher(current_epoch)
                optimizer.step()
                scheduler.step()
                elapsed = time.perf_counter() - step_start
                step += 1
                if index + 1 == batches_per_epoch:
                    epoch = current_epoch + 1
                if step == 1 or step % cfg.training.loop.log_interval_steps == 0 or step == stop_step:
                    logger.log(to_log_scalars({"train/" + key: value for key, value in metrics.items()}
                                             | {"train/lr": scheduler.get_last_lr()[0], "train/epoch": current_epoch,
                                                "train/step_time": elapsed, "train/data_wait_time": data_wait}), step)
                if step >= stop_step or stop_requested:
                    break
                step_start = time.perf_counter()
            if stop_requested:
                break
            if epoch > current_epoch:
                if epoch % cfg.training.val_every_epochs == 0 and validation is not None:
                    metrics = evaluate(
                        model, validation, cfg.dataloader.batch_size, cfg.training.device,
                        stop_requested=lambda: stop_requested,
                    )
                    if metrics is not None:
                        logger.log(to_log_scalars(metrics), step)
                if epoch % cfg.training.save_every_epochs == 0 and step < stop_step and not stop_requested:
                    save_run(model, model.visual, optimizer, scheduler, cfg, epoch, step)
        save_run(model, model.visual, optimizer, scheduler, cfg, epoch, step)
        print(f"[AFRO] Saved encoder.pt and resume.pt at epoch={epoch}, step={step}/{total}")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        close_resources(lambda: close_data_loader(loader), logger.close)


def main():
    train(*load_run("afro", resume_fields={"training.save_every_epochs", "training.val_every_epochs"}))


if __name__ == "__main__":
    main()
