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

# torch.cuda.set_device(0)


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

record_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H_%M_%S'))

RESULT_PATH_VAL = ''


def heru_extract(data_name, num_works, cuda, lr, momentum):
    print("Data loading...")

    # trainloader_inheritable, testloader_inheritable = getDataloader_imagenet_inheritable(args.num_works_tt, args.batch_size, subtask_classes_num=5, num_imgs_per_cate=200)

    print("Model constructing...")
    model = Netwider(13)
    model.printf()

    for task in range(num_works):

        print("LL_Task {} begins !".format(task))

        start = time.time()

        if task != 0:
            model_ = copy.deepcopy(model)
            del model
            model = model_
            model.wider(task - 1)
            model.printf()

        if cuda:
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_epoch = 0
        if data_name == 'cifar100':
            snapshot = './imagenet_exp/outputs/i_inheritable_cifar100_wEWC/record_{0}/Task_{1}'.format(record_time, task)
            snapshot_model = './imagenet_exp/val_outputs/Val_lifelong_scratch_cifar/2023-12-08_07:48:30/task_{0}'.format(task)
        if data_name == 'imagenet100':
            snapshot = './imagenet_exp/outputs/i_inheritable_random_trainnum200_wEWC/record_{0}/Task_{1}'.format(record_time, task)
            snapshot_model = './imagenet_exp/val_outputs//Val_lifelong_scratch_cifar_imagenet/task_{0}'.format(task)

        # load collective-model

        if not os.path.isdir(snapshot):
            print("Building snapshot file: {0}".format(snapshot))
            os.makedirs(snapshot)

        checkpoint_path = os.path.join(snapshot_model, 'checkpoint.pth')

        if os.path.isfile(checkpoint_path):
            print("loading success")
            checkpoint = torch.load(checkpoint_path)
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])

        train_name = 'LL_Train_' + str(task)
        test_name = 'LL_Test_' + str(task)

        print("LL_Task {0} finished ! ".format(task))

        if task == 20:

            layers = model.get_layers_19_20()

            # torch.save(layers.weight, 'extracted_weights.pth')
            return layers

