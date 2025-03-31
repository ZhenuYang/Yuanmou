import os
import datetime
import numpy as np
import time
from samplers import RASampler

import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

import utils
import models
from params import args
from logger import logger
from datasets import build_dataset
from networks.trans_layer import get_trans_layer
from engine import train_one_epoch_scratch, evaluate
from pretrain import create_LearngeneInstances
from networks.learngenepool import LearngenePool
from engine import initialize_model_stitching_layer, evaluate, evaluate_ours

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings

warnings.filterwarnings("ignore")


# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
@torch.no_grad()
def throughput(data_loader, model):
    model.eval()
    # update latency level
    if hasattr(model, 'module'):
        model.module.reset_latency_level(args.latency_level)
    else:
        model.reset_latency_level(args.latency_level)

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

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
    instances = create_LearngeneInstances(blk_length=args.blk_length, init_mode=None)
    model = LearngenePool(instances)
    model.to(device)

    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>Learngene Pool<<<<<<<<<<<<<<<<<<<<<<<<<')
    # logger.info(str(model))

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
        if epoch in [0, 145, 146, 147, 148, 149]:
            ckpt_filename = '{}epoch_checkpoint.pth'.format(epoch)
            utils.save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, args, is_best, output_dir=args.output_dir, filename=ckpt_filename)
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
