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
from engine import train_one_epoch_distill, evaluate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"


def main():
    print(args)
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

    # 定义教师网络
    teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,)
    # 定义学生网络
    student_model = create_model(
            args.student_model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,)

    trans_layer_att = get_trans_layer(teacher_model.blocks[0].embed_dim, student_model.blocks[0].embed_dim, loss_pos=args.loss_pos)
    trans_layer_map = get_trans_layer(teacher_model.blocks[0].embed_dim, student_model.blocks[0].embed_dim, loss_pos=args.loss_pos)
    
    teacher_model.to(device)
    student_model.to(device)
    trans_layer_att.to(device)
    trans_layer_map.to(device)
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>teacher model<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(teacher_model))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>student model<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(student_model))

    if args.distributed:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_model = teacher_model.module

        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu], find_unused_parameters=True)
        student_model = student_model.module

    optimizer = create_optimizer(args, student_model)
    loss_scaler = NativeScaler()  # 相当于loss.backward(create_graph=create_graph) 和 optimizer.step()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    MSE_loss = torch.nn.MSELoss()
    CE_loss = torch.nn.CrossEntropyLoss()
    
    if args.resume:
        logger.info(f'resume from {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            ckpt_path = os.path.join(args.resume, 'checkpoint.pth')
            trans_att_path = os.path.join(args.resume, 'TransLayerAtt_checkpoint.pth')
            trans_map_path = os.path.join(args.resume, 'TransLayerMap_checkpoint.pth')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            trans_att = torch.load(trans_att_path)
            trans_map = torch.load(trans_map_path)
        trans_layer_att.load_state_dict(trans_att['model'])
        trans_layer_map.load_state_dict(trans_map['model'])
        student_model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    start_time = time.time()
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>Parameters<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(str(args))

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    #####################################Training#####################################
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch_distill(
            teacher_model, student_model, trans_layer_att, trans_layer_map,
            MSE_loss, CE_loss, soft_cross_entropy, args.temperature, args.loss_pos, args.alpha, args.distill_loss,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn)
        lr_scheduler.step(epoch)

        # 每个epoch下测试只是为了观察每个epoch下的结果是否有提升，没有其他作用，取最后一个epoch的测试结果为准
        evaluate(data_loader_val, student_model, device, mode='distill')
        is_best = False
        # if epoch == 98:
        #     utils.save_checkpoint({
        #         'model': student_model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'scaler': loss_scaler.state_dict(),
        #         'args': args,
        #     }, args, is_best, output_dir=os.path.join(args.output_dir, '98_epoch'))
        utils.save_checkpoint({
            'model': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(), 
            'args': args,
        }, args, is_best, output_dir=args.output_dir)

        utils.save_checkpoint({
            'model': trans_layer_map.state_dict(),
        }, args, is_best, output_dir=args.output_dir, filename='TransLayerMap_checkpoint.pth')
        utils.save_checkpoint({
            'model': trans_layer_att.state_dict(),
        }, args, is_best, output_dir=args.output_dir, filename='TransLayerAtt_checkpoint.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
