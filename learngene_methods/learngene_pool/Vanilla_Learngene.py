import os
import datetime
import numpy as np
import time
from samplers import RASampler
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from timm.models.vision_transformer import Block, VisionTransformer

import utils
import models
from params import args
from logger import logger
from datasets import build_dataset
from engine import train_one_epoch_scratch, evaluate
from pretrain import create_LearngeneInstances
from networks.learngenepool import LearngenePool
from engine import initialize_model_stitching_layer, evaluate, evaluate_ours


class Descendant(nn.Module):
    def __init__(self, n_low, Ancestry):
        super(Descendant, self).__init__()
        self.n_low = n_low
        self.Ancestry = Ancestry

        self.temp_deit_192 = VisionTransformer(patch_size=16, embed_dim=192, depth=1, num_heads=6, mlp_ratio=4,
                                               qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.temp_deit_768 = VisionTransformer(patch_size=16, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4,
                                               qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # non-learngene_layers
        self.blocks_low = nn.Sequential(*[
            Block(
                dim=192,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for i in range(n_low)]) if n_low > 0 else nn.Identity()
        self.trans_layer = nn.Linear(192, 768) if n_low > 0 else nn.Identity()

        n_high = 3 - n_low
        self.blocks_high = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for i in range(n_high)]) if n_high > 0 else nn.Identity()

        self.non_learngene_layers = nn.Sequential(
            self.blocks_low,
            self.trans_layer,
            self.blocks_high
        )
        # Learngene Layers
        self.learngene_layers = self.Ancestry.blocks[-3:]

    def forward_patch_embed_192(self, x):
        x = self.temp_deit_192.patch_embed(x)
        x = self.temp_deit_192._pos_embed(x)
        return x

    def forward_patch_embed_768(self, x):
        x = self.temp_deit_768.patch_embed(x)
        x = self.temp_deit_768._pos_embed(x)
        return x

    def forward(self, x):
        if self.n_low > 0:
            x = self.forward_patch_embed_192(x)
        elif self.n_low == 0:
            x = self.forward_patch_embed_768(x)
        x = self.non_learngene_layers(x)
        x = self.learngene_layers(x)
        x = self.Ancestry.forward_head(x)

        return x


def main():
    utils.init_distributed_mode(args)
    if utils.get_rank() != 0:
        logger.disabled = True
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, dataset_val, args.nb_classes = build_dataset(is_train=True, args=args)

    if args.distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        # Enabling distributed evaluation
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        # 一种数据增强的办法，同时对输入的图片和标签进行增强
        # Mixup包含两种增强办法，分别是mixup和cutmix, 解析见https://blog.csdn.net/Brikie/article/details/113872771
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 定义learngene_instances
    Ancestry = create_model(
        args.teacher_model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,)
    model = Descendant(args.n_low, Ancestry)
    model.to(device)

    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>Learngene Pool<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(model))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model = model.module

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()  # 相当于loss.backward(create_graph=create_graph) 和 optimizer.step()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    CE_loss = torch.nn.CrossEntropyLoss()

    if args.resume:
        logger.info(f'resume from {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            ckpt_path = os.path.join(args.resume, 'checkpoint.pth')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    start_time = time.time()
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>Parameters<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(args))

    #####################################Training#####################################
    for epoch in range(args.start_epoch, 150):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch_scratch(
            model,
            CE_loss,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn=None,
            cfg_id=args.cfg_id)
        lr_scheduler.step(epoch)

        evaluate(data_loader_val, model, device, mode='scratch', cfg_id=args.cfg_id)
        is_best = False
        utils.save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }, args, is_best, output_dir=args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
