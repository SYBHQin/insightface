import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackAttackEval, CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."


def _dist_is_initialized():
    return distributed.is_available() and distributed.is_initialized()


def _unwrap_ddp(module):
    return module.module if hasattr(module, "module") else module


dist_backend = "nccl" if distributed.is_nccl_available() else "gloo"
try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
if world_size > 1:
    distributed.init_process_group(dist_backend)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@torch.no_grad()
def _get_target_center(module_partial_fc, target_label, device):
    center = torch.zeros(module_partial_fc.embedding_size, device=device)
    local_index = int(target_label) - module_partial_fc.class_start
    if 0 <= local_index < module_partial_fc.num_local:
        center.copy_(module_partial_fc.weight[local_index].detach())
    if _dist_is_initialized():
        distributed.all_reduce(center, op=distributed.ReduceOp.SUM)
    return F.normalize(center.view(1, -1), dim=1)


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        poison_config=_cfg_get(cfg, "poison", None),
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    if _dist_is_initialized():
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
            find_unused_parameters=True)
        if dist_backend == "nccl":
            backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    if hasattr(backbone, "_set_static_graph"):
        backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        _unwrap_ddp(backbone).load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    poison_cfg = _cfg_get(cfg, "poison", None)
    embedding_bd_weight = float(_cfg_get(poison_cfg, "embedding_loss_weight", 0.0))
    poison_target_label = _cfg_get(poison_cfg, "target_label", None)
    if embedding_bd_weight > 0 and poison_target_label is None:
        raise ValueError("config.poison.target_label is required when embedding_loss_weight > 0.")
    if poison_target_label is not None:
        poison_target_label = int(poison_target_label)
        if not 0 <= poison_target_label < cfg.num_classes:
            raise ValueError("config.poison.target_label must be in [0, config.num_classes).")

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_attack_eval = CallBackAttackEval(
        cfg=cfg,
        summary_writer=summary_writer,
        wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, batch in enumerate(train_loader):
            if len(batch) == 3:
                img, local_labels, poison_flags = batch
            else:
                img, local_labels = batch
                poison_flags = None
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            bd_loss = None
            if embedding_bd_weight > 0 and poison_flags is not None:
                poison_mask = poison_flags.bool()
                if torch.any(poison_mask):
                    target_center = _get_target_center(module_partial_fc, poison_target_label, local_embeddings.device)
                    poison_embeddings = F.normalize(local_embeddings[poison_mask], dim=1)
                    bd_loss = 1.0 - torch.sum(poison_embeddings * target_center, dim=1)
                    loss = loss + embedding_bd_weight * bd_loss.mean()

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    log_dict = {
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    }
                    if bd_loss is not None:
                        log_dict['Loss/Backdoor Embedding Loss'] = bd_loss.mean().item()
                    wandb_logger.log(log_dict)
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
                callback_attack_eval(global_step, backbone)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": _unwrap_ddp(backbone).state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(_unwrap_ddp(backbone).state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(_unwrap_ddp(backbone).state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file", default="configs/faces_webface_r50_badnet_onegpu")
    main(parser.parse_args())
