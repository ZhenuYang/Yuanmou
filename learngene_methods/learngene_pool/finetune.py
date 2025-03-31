import os
import time
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
import models
from params import args
from logger import logger
from samplers import RASampler
from datasets import build_dataset
from pretrain import create_learngenepool
from engine import train_one_epoch, evaluate
from networks.learngenepool import Descendant

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = "8"

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

    learngenepool = create_learngenepool(base_mode=True, init=True)

    CE_loss = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>Parameters<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(args))

    #####################################Training#####################################
    for epoch in range(50):
        # randomly select one path in the learngene pool to finetune among 50 epochs
        desc_model = Descendant(args.stitch_id, learngenepool).to(device)
        if args.distributed:
            desc_model = torch.nn.parallel.DistributedDataParallel(desc_model, device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
            desc_model = desc_model.module
        optimizer = create_optimizer(args, desc_model)
        loss_scaler = NativeScaler()  # 相当于loss.backward(create_graph=create_graph) 和 optimizer.step()
        lr_scheduler, _ = create_scheduler(args, optimizer)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            desc_model,
            CE_loss,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn=None)
        lr_scheduler.step(epoch)

        # 每个epoch下测试只是为了观察每个epoch下的结果是否有提升，没有其他作用，取最后一个epoch的测试结果为准
        evaluate(data_loader_val, desc_model, device, mode='scratch')
        is_best = False
        utils.save_checkpoint({
            'model': desc_model.state_dict(),
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