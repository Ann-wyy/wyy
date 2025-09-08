import sys
import os
# 获取父目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import torch
from functools import partial
import torch.distributed as dist
from tqdm import tqdm
from fvcore.common.checkpoint import PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from npz_dataset import NPZDataset
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dinov2_1024.log',  # 指定日志文件名称
    filemode='a'                     # 'a' 表示追加模式，如果文件存在，则在末尾添加日志
)
logger = logging.getLogger(__name__)

def safetensors_to_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt") as ckpt_file:
        for key in ckpt_file.keys():
            state_dict[key] = ckpt_file.get_tensor(key)
    return state_dict

def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )



def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    
    # === 在构建调度器之前，动态计算缩放后的学习率 ===
    global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    
    # 根据你的配置，缩放规则是 "sqrt_wrt_1024"
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        scaled_lr = cfg.optim.base_lr * math.sqrt(global_batch_size / 1024)
    else:
        # Fallback to a default scaling rule if needed, or raise an error
        # 例如，线性缩放:
        # scaled_lr = cfg.optim.base_lr * global_batch_size / 1024
        # 如果缩放规则是未知的，这里最好抛出错误以避免静默失败
        logger.warning(f"Unknown scaling rule: {cfg.optim.scaling_rule}. Using linear scaling.")
        scaled_lr = cfg.optim.base_lr * global_batch_size / 1024
        
    logger.info(f"Calculated scaled LR based on global batch size {global_batch_size}: {scaled_lr}")
    # === 动态计算结束 ===

    lr = dict(
        # 使用计算出的 scaled_lr 作为基准值，而不是配置文件中的 lr: 0.
        base_value=scaled_lr,
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    
    # last_layer_lr_schedule也应该使用缩放后的学习率
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group.get("is_last_layer", False)
        lr_multiplier = param_group.get("lr_multiplier", 1.0)
        wd_multiplier = param_group.get("wd_multiplier", 1.0)
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer,fp16_scaler=model.fp16_scaler, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=cfg.train.saveckp_freq * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )
    best_loss = float('inf')

    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * n_tokens,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader
    data_dir = cfg.dataloader.npz_folder
    dataset = NPZDataset(
        data_dir=data_dir,
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        local_crops_number=cfg.crops.local_crops_number,
    )
    logger.info(f"# of dataset samples: {len(dataset):,d}")
    print(f"Dataset size: {len(dataset)}")
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    # 在主进程上使用 tqdm 封装日志迭代器
    pbar_disable = (distributed.get_global_size() > 1 and not distributed.is_main_process())

    # 将 tqdm 实例赋值给一个变量，例如 `pbar`
    pbar = tqdm(
        metric_logger.log_every(
            data_loader,
            100,  # 日志打印频率
            header,
            max_iter,
            start_iter,
        ),
        total=max_iter,  # 进度条的总长度为总迭代次数
        initial=start_iter,
        disable=pbar_disable,
        mininterval=1.0,
        desc="Training"
    )

    for data in pbar:
        if iteration >= max_iter:
            return

        # 计算当前的 Epoch 信息
        if iteration % OFFICIAL_EPOCH_LENGTH == 0:
            epoch = iteration // OFFICIAL_EPOCH_LENGTH
            pbar.set_description(f"Epoch {epoch + 1}/{cfg.optim.epochs}")

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if distributed.is_main_process():
            # 这里的损失值是当前批次的平均损失
            if losses_reduced < best_loss:
                best_loss = losses_reduced
                
                # 调用 save 函数并传入固定的名称 "best_model"
                checkpointer.save(
                    "best_model",
                    iteration=iteration,
                    best_loss=best_loss,
                    optimizer_state_dict=optimizer.state_dict(),
                    fp16_scaler_state_dict=fp16_scaler.state_dict() if fp16_scaler else None
                )

        # checkpointing
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train(rank, world_size, config):
    try:
        # 你的主训练函数
        is_distributed = world_size > 1
        if is_distributed:
            # dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            logger.info(f"Initialized process group on GPU {rank}.")
        else:
            torch.cuda.set_device(0)
            logger.info(f"Initialized process on Single GPU {rank}.")

        # 构建模型
        model = SSLMetaArch(config).to(torch.device("cuda"))

        student_backbone = model.student.backbone

        # 3. 加载预训练权重
        checkpoint = torch.load("/home/yyi/X_ray/weight/rad_dino.pth", map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        student_backbone.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded pretrained weights into model.student.backbone.")
        model.prepare_for_distributed_training()

        # 执行训练
        do_train(config, model, resume=config.train.resume_from_checkpoint)

    except Exception as e:
        # 在这里捕获所有异常，并使用 logger.exception() 记录
        logger.exception(f"An unexpected error occurred in process {rank}: {e}")
        # 如果需要，可以重新抛出异常，让torchrun知道进程失败了
        # raise
    finally:
        # 确保在任何情况下都销毁进程组，避免警告
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINOv2 training")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--no-resume",action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,)
    args = parser.parse_args()

    # 2. 加载和设置配置
    config = setup(args)

    # 3. 运行训练
    world_size = distributed.get_global_size()
    rank = distributed.get_global_rank()

    train(rank, world_size, config)
