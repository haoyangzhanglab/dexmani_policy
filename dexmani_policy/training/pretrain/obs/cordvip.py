"""CordViP contact/coordination pretraining from xuanxuanzzzii/cordvip.

Commit: ad6d441abe7b0480cf79b0311db266965c1313b6
Sources: train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_pretrain.py
and model/vision/create_fc.py (attributed upstream to SridharPandian/Holo-Dex).
Changes: native arm7+hand12, observation-only data, raw encoder export and resume.
"""

import copy
import math
import signal

import torch
from omegaconf import open_dict
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.interaction.cordvip_encoder import CordViPEncoder
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, set_rng_state, set_seed
from dexmani_policy.training.common.data import close_data_loader
from dexmani_policy.training.common.logging import ExperimentLogger, accumulate_metrics, to_log_scalars
from dexmani_policy.training.common.optimization import get_scheduler
from dexmani_policy.training.pretrain.obs.runtime import evaluate, parse_run, prepare_data, save_run


def create_fc(input_dim, output_dim, hidden_dims):
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
    for index in range(len(hidden_dims) - 1):
        layers.extend((nn.Linear(hidden_dims[index], hidden_dims[index + 1]), nn.ReLU()))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


class CordViPPretrain(nn.Module):
    def __init__(self, n_obs_steps=4, contact_loss_weight=1.0, arm_loss_weight=1.0, hand_loss_weight=1.0):
        super().__init__()
        self.obs_encoder = CordViPEncoder()
        self.n_obs_steps = n_obs_steps
        self.normalizer = LinearNormalizer()
        self.linear_layer_pred_arm = create_fc(n_obs_steps * 2112, 7, [512, 256, 64])
        self.linear_layer_pred_hand = create_fc(n_obs_steps * 2112, 12, [512, 256, 64])
        self.predict_contact_map = create_fc(2048, 1024, [2048, 1024, 1024])
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
    save_every = positive_int(cfg.training.save_every_updates, 'training.save_every_updates')
    set_seed(cfg.training.seed)
    dataset, normalizer, loader = prepare_data(cfg, checkpoint)
    accumulation = cfg.training.loop.gradient_accumulation_steps
    with open_dict(cfg):
        cfg.data_summary = {'samples': len(dataset), 'batches_per_epoch': len(loader),
                            'updates_per_epoch': math.ceil(len(loader) / accumulation)}
    model = CordViPPretrain(**cfg.model)
    model.normalizer = copy.deepcopy(normalizer)
    model.to(cfg.training.device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    total = cfg.training.loop.total_train_steps
    scheduler = get_scheduler(optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps, total)
    epoch = step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        epoch, step = checkpoint['epoch'], checkpoint['global_step']
        set_rng_state(checkpoint['rng_state'], cfg.training.device)
        print(f'[CordViP] Resume epoch {epoch} from its beginning; global_step={step}')
    stop_step = total if max_updates is None else min(total, step + max_updates)
    logger = ExperimentLogger(**cfg.workspace)
    print(f'[CordViP] {dict(cfg.data_summary)}, batch={cfg.dataloader.batch_size}, accumulation={accumulation}, total_updates={total}')
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        logger.save_config(cfg)
        while step < stop_step and not stop_requested:
            loader.generator.manual_seed(cfg.training.seed + epoch)
            model.train()
            current_epoch = epoch
            totals: dict[str, torch.Tensor] = {}
            for index, batch in enumerate(loader):
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
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
                    continue
                gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(gradients).all():
                    raise RuntimeError(f'Nonfinite CordViP gradient at step {step}')
                optimizer.step()
                scheduler.step()
                step += 1
                if index + 1 == len(loader):
                    epoch = current_epoch + 1
                if record:
                    logger.log(to_log_scalars({'train/' + key: value / group_size for key, value in totals.items()}
                                             | {'train/lr': scheduler.get_last_lr()[0], 'train/epoch': current_epoch}), step)
                totals = {}
                if step % save_every == 0:
                    save_run(model, model.obs_encoder, optimizer, scheduler, cfg, epoch, step)
                if step >= stop_step or stop_requested:
                    break
            if epoch > current_epoch:
                if epoch % cfg.training.val_every_epochs == 0:
                    validation = dataset.get_validation_dataset()
                    if validation is not None:
                        logger.log(to_log_scalars(evaluate(model, validation, cfg.dataloader.batch_size, cfg.training.device)), step)
        save_run(model, model.obs_encoder, optimizer, scheduler, cfg, epoch, step)
        print(f'[CordViP] Saved raw encoder.pt and resume.pt at epoch={epoch}, step={step}/{total}')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        close_resources(lambda: close_data_loader(loader), logger.close)


def main():
    train(*parse_run('cordvip'))


if __name__ == '__main__':
    main()
