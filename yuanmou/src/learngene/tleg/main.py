import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path

# 从timm库导入必要的模块
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma

# 从当前tleg模块内部导入依赖
from . import utils
from .datasets import build_dataset
from .engine import train_one_epoch, evaluate
from .losses import DistillationLoss
from .samplers import RASampler
from . import models as tleg_models # 导入所有模型定义，包括 aux_, des_, 和 base models

def main(args):
    """
    TLEG算法的统一训练和评估入口。
    该函数由 ancestor.py 和 descendant.py 调用。
    """
    utils.init_distributed_mode(args)

    print("TLEG 算法配置参数:")
    # 打印配置对象的所有属性
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    device = torch.device(args.DEVICE)

    # --- 设置随机种子 ---
    seed = args.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # --- 数据集和数据加载器 ---
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args.DATASET)
    dataset_val, _ = build_dataset(is_train=False, args=args.DATASET)

    if getattr(args.DISTRIBUTED, 'DISTRIBUTED', True):
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.DISTRIBUTED.DIST_EVAL:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.TRAIN.BATCH_SIZE,
        num_workers=args.DATASET.NUM_WORKERS,
        pin_memory=args.DATASET.PIN_MEM,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.TRAIN.BATCH_SIZE),
        num_workers=args.DATASET.NUM_WORKERS,
        pin_memory=args.DATASET.PIN_MEM,
        drop_last=False
    )

    # --- Mixup 数据增强 ---
    mixup_fn = None
    mixup_active = args.TRAIN.MIXUP > 0 or args.TRAIN.CUTMIX > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.TRAIN.MIXUP, cutmix_alpha=args.TRAIN.CUTMIX,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=args.TRAIN.SMOOTHING, num_classes=args.nb_classes)

    # --- 创建模型 ---
    print(f"创建模型: {args.MODEL.TYPE}")
    model = create_model(
        args.MODEL.TYPE,
        pretrained=args.MODEL.PRETRAINED,
        num_classes=args.nb_classes,
        drop_rate=args.MODEL.DROP,
        drop_path_rate=args.MODEL.DROP_PATH,
    )
    
    # --- 关键逻辑：加载“学习基因” ---
    load_gene_path = getattr(args.MODEL, 'LOAD_GENE', '')
    if load_gene_path:
        print(f"正在从基因文件加载权重: {load_gene_path}")
        checkpoint = torch.load(load_gene_path, map_location='cpu')
        
        checkpoint_model = checkpoint.get('model', checkpoint)
        
        # 获取后代模型的层数
        num_layers = model.get_num_layers() if hasattr(model, 'get_num_layers') else len(getattr(model, 'blocks', []))
        if num_layers == 0:
            raise ValueError("无法确定后代模型的层数 (model.blocks不存在或为空)")

        print(f"后代模型深度为: {num_layers} 层")
        
        # 调用基因转换函数
        checkpoint_new = utils.gene_tranfer(checkpoint_model, num_layers, load_head=False)
        
        # 加载转换后的权重
        msg = model.load_state_dict(checkpoint_new, strict=False)
        print("基因权重加载和转换完成。加载信息:", msg)


    model.to(device)

    model_ema = None
    if args.MODEL.EMA:
        model_ema = ModelEma(model, decay=args.MODEL.EMA_DECAY, device='cpu', resume='')

    model_without_ddp = model
    if getattr(args.DISTRIBUTED, 'DISTRIBUTED', True):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.DISTRIBUTED.gpu])
        model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数量: {n_parameters / 1e6:.2f} M')

    # --- 优化器和学习率调度器 ---
    if not args.TRAIN.UNSCALE_LR:
        linear_scaled_lr = args.TRAIN.LR * args.TRAIN.BATCH_SIZE * utils.get_world_size() / 512.0
        args.TRAIN.LR = linear_scaled_lr
        
    optimizer = create_optimizer(args.TRAIN, model_without_ddp)
    loss_scaler = utils.NativeScaler()
    lr_scheduler, _ = create_scheduler(args.TRAIN, optimizer)

    # --- 损失函数 ---
    if mixup_fn:
        criterion = SoftTargetCrossEntropy()
    elif args.TRAIN.SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.TRAIN.SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Distillation loss (如果配置了teacher model)
    # 此处为简化，暂不实现，但保留扩展性
    criterion = DistillationLoss(criterion, None, 'none', 0, 0)
    
    # --- 训练和评估 ---
    output_dir = Path(args.OUTPUT_DIR)
    if args.OUTPUT_DIR:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始训练，共 {args.TRAIN.EPOCHS} 个 epochs")
    start_time = time.time()
    max_accuracy = 0.0
        
    for epoch in range(args.TRAIN.START_EPOCH, args.TRAIN.EPOCHS):
        if getattr(args.DISTRIBUTED, 'DISTRIBUTED', True):
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, model_ema=model_ema, mixup_fn=mixup_fn,
        )

        lr_scheduler.step(epoch)

        if args.OUTPUT_DIR:
            checkpoint_path = output_dir / 'checkpoint.pth'
            if utils.is_main_process():
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }
                torch.save(save_obj, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Epoch {epoch}: 精度 Acc@1: {test_stats['acc1']:.2f}%")
        
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'历史最高精度: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.OUTPUT_DIR and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'训练总时间: {total_time_str}')