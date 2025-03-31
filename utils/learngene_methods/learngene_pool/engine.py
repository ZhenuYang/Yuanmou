"""
Train and eval functions used in main.py
"""
import math
import os.path
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import utils
from logger import logger
import json
from utils import save_on_master_eval_res

from fvcore.nn import FlopCountAnalysis


def initialize_model_stitching_layer(model, mixup_fn, data_loader,  device, mode):
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        model.initialize_stitching_weights(samples, mode)

        break


def train_one_epoch(model: torch.nn.Module,
                    criterion,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():  # 使用混合精度以降低内存开销
            desc_out = model(samples)
            loss = criterion(samples, desc_out, targets)  # 分类损失
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=is_second_order)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_scratch(model: torch.nn.Module,
                    CE_loss,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    cfg_id=0,):
    model.train()
    # model.reset_stitch_id(cfg_id)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():  # 使用混合精度以降低内存开销
            desc_out = model(samples)
            cls_loss = CE_loss(desc_out, targets)  # 分类损失

            total_loss = cls_loss
        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=is_second_order)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_distill(tearcher_model: torch.nn.Module, student_model: torch.nn.Module,
                    trans_layer_att, trans_layer_map,
                    MSE_loss, CE_loss, soft_cross_entropy, temperature, loss_pos, alpha, distill_loss,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler, 
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None):
    student_model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        att_loss = 0.
        rep_loss = 0.

        with torch.cuda.amp.autocast():  # 使用混合精度以降低内存开销
            stu_out, stu_out_atts, stu_out_maps, stu_out_embed = student_model.forward_features(samples)
            with torch.no_grad():
                tea_out, tea_out_atts, tea_out_maps, tea_out_embed = tearcher_model.forward_features(samples)

            num_stu_blk = len(stu_out_atts)
            
            cls_loss = CE_loss(stu_out, targets)  # 分类损失
            logits_loss = soft_cross_entropy(stu_out/temperature, tea_out/temperature) # 全连接层输出的特征损失
            if loss_pos == '[end]':
                tea_out_att = trans_layer_att[0](tea_out_atts[-1])
                tea_out_map = trans_layer_map[0](tea_out_maps[-1])
                att_loss = MSE_loss(stu_out_atts[-1], tea_out_att)  # 最后一个blk中Attention层输出特征的损失
                rep_loss = MSE_loss(stu_out_maps[-1], tea_out_map)  # 最后一个blk输出特征的损失

            elif loss_pos == '[mid,end]':
                stu_point = [int((num_stu_blk/3)*2-1), int((num_stu_blk/3)*3-1)]
                tea_point = [7, 11]
                for id, i, j in zip([0, 1], stu_point, tea_point):
                    tea_out_att = trans_layer_att[id](tea_out_atts[j])
                    tea_out_map = trans_layer_map[id](tea_out_maps[j])
                    att_loss += MSE_loss(stu_out_atts[i], tea_out_att)
                    rep_loss += MSE_loss(stu_out_maps[i], tea_out_map)

            elif loss_pos == '[front,end]':
                stu_point = [int((num_stu_blk/3)-1), int((num_stu_blk/3)*2-1), int((num_stu_blk/3)*3-1)]
                tea_point = [3, 7, 11]
                for id, i, j in zip([0,1,2], stu_point, tea_point):
                    tea_out_att = trans_layer_att[id](tea_out_atts[j])
                    tea_out_map = trans_layer_map[id](tea_out_maps[j])
                    att_loss += MSE_loss(stu_out_atts[i], tea_out_att)
                    rep_loss += MSE_loss(stu_out_maps[i], tea_out_map)
            

            if distill_loss == 'logits':
                alpha = 0.5
                total_loss = alpha * cls_loss + (1 - alpha) * (logits_loss)
            elif distill_loss == 'logits+rep':
                alpha = 0.5
                total_loss = alpha * cls_loss + (1 - alpha) * (logits_loss + rep_loss)
            elif distill_loss == 'all':
                total_loss = alpha * cls_loss + (1 - alpha) * (logits_loss + rep_loss + att_loss)

        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm, parameters=student_model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, mode='scratch', cfg_id=0):
    CE_loss = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # model.reset_stitch_id(cfg_id)

    print_freq = len(data_loader)-1
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if mode == 'scratch':
                out = model(images)
            elif mode == 'distill':
                out, _, _, _ = model.forward_features(images)
            loss = CE_loss(out, target)

        acc1, acc5 = accuracy(out, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_ours(data_loader, model, device, output_dir, blk_length):
    # check last config:
    last_cfg_id = -1
    if os.path.exists(output_dir):
        with open(output_dir, 'r') as f:
            for line in f.readlines():
                epoch_stat = json.loads(line.strip())
                last_cfg_id = epoch_stat['cfg_id']

    criterion = torch.nn.CrossEntropyLoss()

    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if hasattr(model, 'module'):
        num_configs = model.module.num_configs
    else:
        num_configs = model.num_configs

    # for cfg_id in range(last_cfg_id+1, num_configs):
    if blk_length == 6:
        cfg_ids = [0, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # blk_length=6
    if blk_length == 9:
        cfg_ids = [0, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                      51, 52]  # blk_length=9
    for cfg_id in cfg_ids:
        if hasattr(model, 'module'):
            model.module.reset_stitch_id(cfg_id)
        else:
            model.reset_stitch_id(cfg_id)

        logger.info(f'------------- Evaluting stitch config {cfg_id}/{num_configs} -------------')

        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).cuda())
        flops = flops.total()
        converted = flops / 1e9
        converted = round(converted, 2)
        logger.info(f'FLOPs = {converted}')

        metric_logger = utils.MetricLogger(delimiter="  ")

        print_freq = len(data_loader) - 1
        for images, target in metric_logger.log_every(data_loader, print_freq, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info('cfg_id = ' + str(
            cfg_id) + '  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                    .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        log_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats['cfg_id'] = cfg_id
        log_stats['flops'] = flops
        log_stats['params'] = model.get_model_size(cfg_id)
        save_on_master_eval_res(log_stats, output_dir)

