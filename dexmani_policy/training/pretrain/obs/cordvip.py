"""CordViP contact/coordination pretraining from xuanxuanzzzii/cordvip.

Commit: ad6d441abe7b0480cf79b0311db266965c1313b6
Source: train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_pretrain.py
Changes: native arm7+hand12, observation-only data, raw encoder export and resume.
"""

import signal
import time

import hydra
import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.interaction.correspondence.contact_coordination_encoder import (
    ContactCoordinationEncoder,
)
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import build_mlp
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, set_seed
from dexmani_policy.training.common.data import build_train_loader, close_data_loader, epoch_position
from dexmani_policy.training.common.logging import ExperimentLogger, accumulate_metrics, to_log_scalars
from dexmani_policy.training.common.optimization import get_scheduler
from dexmani_policy.training.pretrain.obs.common import evaluate, load_run, restore_run, save_run


class CordViPPretrain(nn.Module):
    def __init__(self, normalization, n_obs_steps=4, contact_loss_weight=1.0, arm_loss_weight=1.0, hand_loss_weight=1.0):
        super().__init__()
        self.obs_encoder = ContactCoordinationEncoder()
        self.n_obs_steps = n_obs_steps
        shapes = {'hand_pointcloud': (3,), 'object_pointcloud': (3,), 'joint_state': (19,)}
        self.normalizer = LinearNormalizer({key: shape for key, shape in shapes.items() if normalization[key] != 'identity'})
        self.linear_layer_pred_arm = build_mlp(
            [n_obs_steps * 2112, 512, 256, 64, 7], activation=nn.ReLU, normalize=False,
        )
        self.linear_layer_pred_hand = build_mlp(
            [n_obs_steps * 2112, 512, 256, 64, 12], activation=nn.ReLU, normalize=False,
        )
        self.predict_contact_map = build_mlp(
            [2048, 2048, 1024, 1024, 1024], activation=nn.ReLU, normalize=False,
        )
        self.contact_loss_weight = contact_loss_weight
        self.arm_loss_weight = arm_loss_weight
        self.hand_loss_weight = hand_loss_weight

    def forward(self, batch):
        obs = {key: self.normalizer[key].normalize(value) if key in self.normalizer.fields else value
               for key, value in batch['obs'].items()}
        robot_pc, object_pc, hand_state, arm_state = self.obs_encoder(obs)
        contact = self.predict_contact_map(torch.cat((robot_pc, object_pc), dim=-1))
        hand_condition = torch.cat((robot_pc.flatten(1), object_pc.flatten(1), hand_state.flatten(1)), dim=1)
        arm_condition = torch.cat((robot_pc.flatten(1), object_pc.flatten(1), arm_state.flatten(1)), dim=1)
        arm = self.linear_layer_pred_arm(hand_condition)
        hand = self.linear_layer_pred_hand(arm_condition)
        target = obs['joint_state'][:, self.n_obs_steps - 1]
        contact_loss = F.mse_loss(contact, batch['contact_map'])
        arm_loss = F.mse_loss(arm, target[:, :7])
        hand_loss = F.mse_loss(hand, target[:, 7:19])
        loss = self.arm_loss_weight * arm_loss + self.hand_loss_weight * hand_loss + self.contact_loss_weight * contact_loss
        return loss, {'loss': loss.detach(), 'contact': contact_loss.detach(),
                      'arm': arm_loss.detach(), 'hand': hand_loss.detach()}


def train(cfg, checkpoint=None, max_updates=None):
    if cfg.dataset.obs_horizon != cfg.model.n_obs_steps:
        raise ValueError('CordViP dataset and model must use matching observation steps')
    positive_int(cfg.training.save_every_updates, 'training.save_every_updates')
    positive_int(cfg.training.val_every_epochs, 'training.val_every_epochs')
    set_seed(cfg.training.seed)
    dataset = hydra.utils.instantiate(cfg.dataset)
    cfg.dataset.split_manifest = dataset.split_manifest
    loader = build_train_loader(cfg, dataset)
    batches_per_epoch = loader.batch_sampler.batches_per_epoch
    accumulation = cfg.training.loop.gradient_accumulation_steps
    model = CordViPPretrain(cfg.normalization, **cfg.model)
    if checkpoint is None:
        model.normalizer.load_state_dict(dataset.get_normalizer(cfg.normalization).state_dict(), strict=True)
    model.to(cfg.training.device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    total = cfg.training.loop.total_train_steps
    scheduler = get_scheduler(optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps, total)
    step = 0
    if checkpoint is not None:
        step = restore_run(model, optimizer, scheduler, checkpoint, device=cfg.training.device)
    epoch, start_batch = epoch_position(step, batches_per_epoch, accumulation)
    if checkpoint is not None:
        print(f"Resume epoch={epoch}, batch={start_batch}, global_step={step}")
    stop_step = total if max_updates is None else min(total, step + max_updates)
    logger = ExperimentLogger(**cfg.workspace)
    print(f'[CordViP] samples={len(dataset)}, batches_per_epoch={batches_per_epoch}, '
          f'batch={cfg.dataloader.batch_size}, accumulation={accumulation}, total_updates={total}')
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        logger.save_config(cfg)
        while step < stop_step and not stop_requested:
            epoch, start_batch = epoch_position(step, batches_per_epoch, accumulation)
            loader.batch_sampler.set_epoch(epoch, start_batch)
            model.train()
            current_epoch = epoch
            totals: dict[str, torch.Tensor] = {}
            step_start = data_wait_start = time.perf_counter()
            data_wait = 0.0
            for index, batch in enumerate(loader, start=start_batch):
                data_wait += time.perf_counter() - data_wait_start
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, batches_per_epoch - group_start)
                if index == group_start:
                    optimizer.zero_grad(set_to_none=True)
                batch = dict_apply(batch, lambda x: x.to(cfg.training.device, non_blocking=True))
                loss, metrics = model(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite CordViP loss at step {step}')
                (loss / group_size).backward()
                record = step == 0 or (step + 1) % cfg.training.loop.log_interval_steps == 0 or step + 1 == stop_step
                if record:
                    accumulate_metrics(totals, metrics)
                if index - group_start + 1 != group_size:
                    data_wait_start = time.perf_counter()
                    continue
                gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(gradients).all():
                    raise RuntimeError(f'Nonfinite CordViP gradient at step {step}')
                optimizer.step()
                scheduler.step()
                elapsed = time.perf_counter() - step_start
                step += 1
                if index + 1 == batches_per_epoch:
                    epoch = current_epoch + 1
                if record:
                    logger.log(to_log_scalars({'train/' + key: value / group_size for key, value in totals.items()}
                                             | {'train/lr': scheduler.get_last_lr()[0], 'train/epoch': current_epoch,
                                                'train/step_time': elapsed, 'train/data_wait_time': data_wait}), step)
                totals = {}
                if step >= stop_step or stop_requested:
                    break
                if step % cfg.training.save_every_updates == 0:
                    save_run(model, model.obs_encoder, optimizer, scheduler, cfg, epoch, step)
                step_start = data_wait_start = time.perf_counter()
                data_wait = 0.0
            if stop_requested:
                break
            if epoch > current_epoch:
                if epoch % cfg.training.val_every_epochs == 0:
                    validation = dataset.get_validation_dataset()
                    if validation is not None:
                        metrics = evaluate(
                            model, validation, cfg.dataloader.batch_size, cfg.training.device,
                            stop_requested=lambda: stop_requested,
                        )
                        if metrics is not None:
                            logger.log(to_log_scalars(metrics), step)
        save_run(model, model.obs_encoder, optimizer, scheduler, cfg, epoch, step)
        print(f'[CordViP] Saved raw encoder.pt and resume.pt at epoch={epoch}, step={step}/{total}')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        close_resources(lambda: close_data_loader(loader), logger.close)


def main():
    train(*load_run('cordvip', resume_fields={'training.save_every_updates', 'training.val_every_epochs'}))


if __name__ == '__main__':
    main()
