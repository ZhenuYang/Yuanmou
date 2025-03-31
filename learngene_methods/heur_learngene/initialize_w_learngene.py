import argparse
import time
import datetime
import sys

sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random
import warnings
import xlwt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable

from utils.train import train, test, train_ewc, test_ewc, train_ewc_vgg, test_ewc_vgg

from utils.models.models import vgg_compression_ONE
# from utils.imagenetDataloader import getDataloader_imagenet_inheritable
from utils.network_wider import Netwider
from learngene.dataloader.dataloader import get_inheritable_heur
from extract_learngene import heru_extract

# torch.cuda.set_device(0)

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

record_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H_%M_%S'))

RESULT_PATH_VAL = ''


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print('making dir: %s' % output_dir)

    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def heru_initialize(data_name, path, batch_size = 32, epochs = 100, lr = 0.01, momentum = 0.9, no_cuda = False,
                    seed = 1, num_works = 21, num_works_tt = 5, num_imgs_per_cat_train = 20):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # cifar100
    inherit_path = path
    trainloader_inheritable, testloader_inheritable = get_inheritable_heur(data_name, num_works_tt, batch_size, num_imgs_per_cate=num_imgs_per_cat_train, path=inherit_path)

    layers = heru_extract(data_name, num_works, cuda, lr, momentum)
    if data_name == 'cifar100':
        snapshot = './imagenet_exp/outputs/i_inheritable_cifar100_wEWC/record_{0}/Task_{1}'.format(record_time, 20)
    if data_name == 'imagenet100':
        snapshot = './imagenet_exp/outputs/i_inheritable_random_trainnum200_wEWC/record_{0}/Task_{1}'.format(record_time, 20)

    model_vgg = vgg_compression_ONE(layers, 2048, num_class=5)
    model_v = copy.deepcopy(model_vgg)
    del model_vgg
    model_vgg = model_v
    # model_vgg.printf()
    # model_vgg = model_vgg.cuda()

    for task_tt in range(num_works_tt):

        print("TT_Task {} begins !".format(task_tt))

        train_tt_name = 'TT_Train_200_' + str(task_tt)
        test_tt_name = 'TT_Test_200_' + str(task_tt)

        sheet_task = book.add_sheet('TT_Task_200_{0}'.format(task_tt), cell_overwrite_ok=True)
        cnt_epoch = 0

        for epoch in range(epochs):
            optimizer = optim.SGD(model_vgg.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

            criterion = nn.CrossEntropyLoss()

            train_loss, train_acc = train_ewc(trainloader_inheritable[task_tt](epoch), model_vgg, criterion,
                                              optimizer, epoch, cuda, \
                                              snapshot=snapshot, name=train_tt_name)

            test_loss, test_acc = test_ewc(testloader_inheritable[task_tt](epoch), model_vgg, criterion,
                                           optimizer, epoch, cuda, \
                                           snapshot=snapshot, name=test_tt_name)

            sheet_task.write(cnt_epoch, 0, 'Epoch_{0}'.format(cnt_epoch))
            sheet_task.write(cnt_epoch, 1, train_loss)
            sheet_task.write(cnt_epoch, 2, train_acc.item())
            sheet_task.write(cnt_epoch, 3, test_loss)
            sheet_task.write(cnt_epoch, 4, test_acc.item())

            cnt_epoch = cnt_epoch + 1

        del model_vgg
        model_vgg = model_v
        # model_vgg.printf()
        model_vgg = model_vgg.cuda()

        print("TT_Task {0} finished !".format(task_tt))

    if data_name == 'cifar100':
        book.save(r'./imagenet_exp/outputs/i_inheritable_random_trainnum200_wEWC/i_inheritable_cifar100_wEWC.xls')
    if data_name == 'imagenet100':
        book.save(r'./imagenet_exp/outputs/i_inheritable_random_trainnum200_wEWC/i_inheritable_random_trainnum200_wEWC.xls')

heru_initialize('cifar100', r'D:\stu_res\reconstruct\YuanMou\learngene\datasets\exp_data\data_cifar100\2023-12-08_07_41_26\inheritabledataset')