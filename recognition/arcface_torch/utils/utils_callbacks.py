import logging
import os
import time
from typing import List

import torch

from attack_eval import build_trigger_from_overrides, evaluate_backdoor, normalize_source_labels
from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


def _get_rank_world_size():
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_rank(), distributed.get_world_size()
    return 0, 1


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112), wandb_logger=None):
        self.rank, _ = _get_rank_world_size()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            if self.summary_writer is not None:
                self.summary_writer: SummaryWriter
                self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )
            if self.wandb_logger:
                import wandb
                self.wandb_logger.log({
                    f'Acc/val-Acc1 {self.ver_name_list[i]}': acc1,
                    f'Acc/val-Acc2 {self.ver_name_list[i]}': acc2,
                    # f'Acc/val-std1 {self.ver_name_list[i]}': std1,
                    # f'Acc/val-std2 {self.ver_name_list[i]}': acc2,
                })

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackAttackEval(object):

    def __init__(self, cfg, summary_writer=None, wandb_logger=None):
        self.rank, _ = _get_rank_world_size()
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

        attack_cfg = _cfg_get(cfg, "attack_eval", None)
        poison_cfg = _cfg_get(cfg, "poison", None)

        self.enabled = bool(_cfg_get(attack_cfg, "enabled", False))
        if not self.enabled:
            return

        target_label = _cfg_get(poison_cfg, "target_label", None)
        if target_label is None:
            raise ValueError("config.poison.target_label is required when config.attack_eval.enabled = True.")

        self.rec_root = _cfg_get(attack_cfg, "rec", None) or cfg.rec
        self.training_rec_root = cfg.rec
        self.target_label = int(target_label)
        self.template_count = int(_cfg_get(attack_cfg, "template_count", 20))
        self.target_test_count = int(_cfg_get(attack_cfg, "target_test_count", 20))
        self.calib_count = int(_cfg_get(attack_cfg, "calib_count", 5000))
        self.probe_count = int(_cfg_get(attack_cfg, "probe_count", 2000))
        self.batch_size = int(_cfg_get(attack_cfg, "batch_size", 128))
        self.far = float(_cfg_get(attack_cfg, "far", 1e-3))
        self.seed = int(_cfg_get(attack_cfg, "seed", 2048))
        self.frequent = int(_cfg_get(attack_cfg, "verbose", _cfg_get(cfg, "verbose", 0)))
        self.trigger = build_trigger_from_overrides(poison_cfg)
        self.source_labels = normalize_source_labels(
            _cfg_get(poison_cfg, "source_labels", None),
            "config.poison.source_labels",
        )

        if self.frequent <= 0:
            raise ValueError("config.attack_eval.verbose must be a positive integer when attack evaluation is enabled.")

    def __call__(self, num_update, backbone: torch.nn.Module):
        if not self.enabled or self.rank != 0 or num_update <= 0 or num_update % self.frequent != 0:
            return

        backbone.eval()
        try:
            metrics = evaluate_backdoor(
                backbone=backbone,
                rec_root=self.rec_root,
                target_label=self.target_label,
                template_count=self.template_count,
                target_test_count=self.target_test_count,
                calib_count=self.calib_count,
                probe_count=self.probe_count,
                batch_size=self.batch_size,
                far=self.far,
                seed=self.seed,
                trigger=self.trigger,
                source_labels=self.source_labels,
                training_rec_root=self.training_rec_root,
                strict_counts=False,
            )
        except Exception as exc:
            logging.warning("[attack_eval][%d]Skipped attack evaluation: %s", num_update, exc)
            backbone.train()
            return
        logging.info(
            "[attack_eval][%d]threshold@FAR=%1.6f clean_far=%1.6f target_tpr=%1.6f attack_asr=%1.6f",
            num_update,
            metrics["threshold"],
            metrics["clean_far"],
            metrics["target_tpr"],
            metrics["attack_asr"],
        )
        for warning in metrics["warnings"]:
            logging.warning("[attack_eval][%d]%s", num_update, warning)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar("attack_eval/clean_far", metrics["clean_far"], num_update)
            self.summary_writer.add_scalar("attack_eval/target_tpr", metrics["target_tpr"], num_update)
            self.summary_writer.add_scalar("attack_eval/attack_asr", metrics["attack_asr"], num_update)
            self.summary_writer.add_scalar("attack_eval/threshold", metrics["threshold"], num_update)

        if self.wandb_logger:
            self.wandb_logger.log({
                "Acc/attack-clean_far": metrics["clean_far"],
                "Acc/attack-target_tpr": metrics["target_tpr"],
                "Acc/attack-asr": metrics["attack_asr"],
                "Acc/attack-threshold": metrics["threshold"],
            })

        backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank, self.world_size = _get_rank_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
