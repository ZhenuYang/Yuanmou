import os
import datetime
import numpy as np
import time
from samplers import RASampler

import torch
import torch.nn as nn
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
from engine import evaluate

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings

warnings.filterwarnings("ignore")

def init_LearngenePool(learngenepool, blk_length, init_mode):
    init_learngenepool = []
    if blk_length == 6:
        if init_mode == 'ours':
            ckpt_paths = [
                './train_results/distill/deit_base_patch16_224-deit_tiny6_patch16_224/LearngenePool_[front,end]/checkpoint.pth',
                './train_results/distill/deit_base_patch16_224-deit_small6_patch16_224/LearngenePool_[end]/checkpoint.pth',
                './train_results/distill/deit_base_patch16_224-deit_base6_patch16_224/LearngenePool_[end]/checkpoint.pth']
        if init_mode == 'snnet':
            ckpt_paths = [
                './train_results/scratch/tiny6/checkpoint.pth',  # instance.0.
                './train_results/scratch/small6/checkpoint.pth',  # instance.1.
                './train_results/scratch/base6/checkpoint.pth']  # instance.2.
    if blk_length == 9:
        if init_mode == 'ours':
            ckpt_paths = [
                './train_results/distill/deit_base_patch16_224-deit_tiny9_patch16_224/LearngenePool_[front,end]/checkpoint.pth',
                './train_results/distill/deit_base_patch16_224-deit_small9_patch16_224/LearngenePool_[end]/checkpoint.pth',
                './train_results/distill/deit_base_patch16_224-deit_base9_patch16_224/LearngenePool_[end]/checkpoint.pth']
        if init_mode == 'snnet':
            ckpt_paths = [
                './train_results/scratch/tiny9/checkpoint.pth',
                './train_results/scratch/small9/checkpoint.pth',
                './train_results/scratch/base9/checkpoint.pth']

    for id, (model, ckpt_path) in enumerate(zip(learngenepool, ckpt_paths)):
        if ckpt_path == './train_results/scratch/small6/checkpoint.pth' or ckpt_path == './train_results/scratch/small9/checkpoint.pth':
            init_learngenepool.append(model)
            continue
        param_dict = torch.load(ckpt_path, map_location='cpu')['model']

        if ckpt_path == './train_results/scratch/tiny9/checkpoint.pth':
            # assert id == 0
            new_param_dict = {}
            key = 'subnetworks.{}'.format(2)
            for k, v in param_dict.items():
                if key in k:
                    new_key = k[14:]
                    # new_key = k
                    new_param_dict[new_key] = v
            model.load_state_dict(new_param_dict)
            init_learngenepool.append(model)
            continue

        if init_mode == 'snnet':
            new_param_dict = {}
            # model_dict = model.state_dict()
            key = 'instances.{}'.format(id)
            for k, v in param_dict.items():
                if key in k:
                    new_key = k[12:]
                    new_param_dict[new_key] = v
            # assert model_dict.keys() == new_param_dict.keys()
            # model_dict.update(new_param_dict)
            model.load_state_dict(new_param_dict)
        else:
            model.load_state_dict(param_dict)
        init_learngenepool.append(model)
    logger.info('The Learngene Pool is been initialized by mode {}'.format(init_mode))
    return init_learngenepool


def create_LearngeneInstances(blk_length, init_mode='scratch'):
    LearngeneInstances = []
    if blk_length == 6:
        deit_base6 = create_model(
            'deit_base6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_small6 = create_model(
            'deit_small6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_tiny6 = create_model(
            'deit_tiny6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        LearngeneInstances = [deit_tiny6, deit_small6, deit_base6]
    elif blk_length == 9:
        deit_base9 = create_model(
            'deit_base9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_small9 = create_model(
            'deit_small9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_tiny9 = create_model(
            'deit_tiny9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        LearngeneInstances = [deit_tiny9, deit_small9, deit_base9]
    if init_mode == 'scratch':
        logger.info('Learngene Instances are trained from scratch!')
        return LearngeneInstances
    LearngeneInstances = init_LearngenePool(LearngeneInstances, blk_length, init_mode)
    return LearngeneInstances


def create_learngenepool(base_mode=False, small_mode=False, tiny_mode=False, init=False):
    learngenepool = []
    ckpt_path = []
    ckpt_root = './train_results/distill'
    if base_mode:
        deit_base3 = create_model(
            'deit_base3_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_base6 = create_model(
            'deit_base6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_base9 = create_model(
            'deit_base9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )

        base_list = [deit_base3, deit_base6, deit_base9]
        learngenepool += base_list

    if small_mode:
        deit_small3 = create_model(
            'deit_small3_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_small6 = create_model(
            'deit_small6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_small9 = create_model(
            'deit_small9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        small_list = [deit_small3, deit_small6, deit_small9]
        # if init:
        #     ckpt_path = [os.path.join(ckpt_root, 'deit_small_path16_224-deit_small3_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth'),
        #                  os.path.join(ckpt_root, 'deit_small_path16_224-deit_small6_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth'),
        #                  os.path.join(ckpt_root, 'deit_small_path16_224-deit_small9_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth')
        #                  ]
        #     small_list = init_learngenepool(small_list, ckpt_path)
        learngenepool += small_list

    if tiny_mode:
        deit_tiny3 = create_model(
            'deit_tiny3_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_tiny6 = create_model(
            'deit_tiny6_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )
        deit_tiny9 = create_model(
            'deit_tiny9_patch16_224',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, )

        tiny_list = [deit_tiny3, deit_tiny6, deit_tiny9]
        # if init:
        #     ckpt_path = [os.path.join(ckpt_root, 'deit_tiny_path16_224-deit_tiny3_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth'),
        #                  os.path.join(ckpt_root, 'deit_tiny_path16_224-deit_tiny6_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth'),
        #                  os.path.join(ckpt_root, 'deit_tiny_path16_224-deit_tiny9_path16_224',
        #                               'LearngenePool_[end]',
        #                               'checkpoint.pth')
        #                  ]
        #     tiny_list = init_learngenepool(tiny_list, ckpt_path)
        learngenepool += tiny_list
    return learngenepool