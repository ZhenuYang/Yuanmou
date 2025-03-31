from __future__ import print_function
import numpy as np
import datetime
import random
import os
import shutil
import sys

sys.path.append("../")
import argparse
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt


parser = argparse.ArgumentParser(description='make_data')

parser.add_argument('--data_name', type=str, default='')
parser.add_argument('--num_imgs_per_cat_train', type=int, default=200)
parser.add_argument('--path', type=str, default='./', help='path of base classes')

args = parser.parse_args()

DATASET_DIR = args.path

_CIFAR_DATASET_DIR = './dataset/Cifar/cifar100'

_IMAGENET_DATASET_DIR = './dataset/Imagenet/Imagenet2012/Data/CLS-LOC'

time_now = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H_%M_%S'))

subtask_all_id = {}

CIFAR100_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
IMAGENET100_TRAIN_MEAN = [0.485, 0.456, 0.406]
IMAGENET100_TRAIN_STD = [0.229, 0.224, 0.225]


class GenericDataset_ll(data.Dataset):

    def __init__(self, dir_name, dataset_name, split, task_iter_item=0, subtask_class_num=5, random_sized_crop=False,
                 num_imgs_per_cat=300, label_division=True):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat

        global subtask_all_id
        global time_now

        if self.dataset_name == 'cifar100':

            self.mean_pix = CIFAR100_TRAIN_MEAN
            self.std_pix = CIFAR100_TRAIN_STD

            if self.split != 'train':
                transforms_list = [
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    lambda x: np.asarray(x),
                ]

            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]

            self.transform = transforms.Compose(transforms_list)
            split_data_dir = DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)

            subtask_all_id[task_iter_item] = random.sample(range(0, 64), subtask_class_num)

            classes = [k for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]

            class_to_idx = {k: v for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}

            imgs = []

            task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'continualdataset', \
                                            'Task_' + str(task_iter_item))

            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)

            with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
                file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, 'continualdata', class_to_idx))

            for c in classes:

                pths = os.path.join(DATASET_DIR + '/' + self.split, c)

                task_class_folder_path = os.path.join(task_folder_path, c)

                if not os.path.exists(task_class_folder_path):
                    os.makedirs(task_class_folder_path)

                all_imgs = os.listdir(pths)

                samples = all_imgs[0:num_imgs_per_cat]

                for sp in samples:
                    shutil.copy(os.path.join(DATASET_DIR + '/' + self.split, c, sp), task_class_folder_path)
                    imgs.append((os.path.join(DATASET_DIR + '/' + self.split, c, sp), class_to_idx[c]))

            self.data = datasets.ImageFolder(task_folder_path, self.transform)

            if self.dataset_name == 'imagenet100':

                self.mean_pix = IMAGENET100_TRAIN_MEAN
                self.std_pix = IMAGENET100_TRAIN_STD

                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]

                self.transform = transforms.Compose(transforms_list)

                split_data_dir = DATASET_DIR + '/train'

                self.data = datasets.ImageFolder(split_data_dir, self.transform)

                subtask_all_id[task_iter_item] = random.sample(range(0, 80), subtask_class_num)

                classes = [k for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]

                class_to_idx = {k: v for (k, v) in self.data.class_to_idx.items() if
                                v in subtask_all_id[task_iter_item]}

                imgs = []

                task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'continualdataset', \
                                                'Task_' + str(task_iter_item))

                if not os.path.exists(task_folder_path):
                    os.makedirs(task_folder_path)

                with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'),
                          'a') as file_val:
                    file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, 'continualdata', class_to_idx))

                for c in classes:

                    pths = os.path.join(DATASET_DIR, 'train', c)

                    task_class_folder_path = os.path.join(task_folder_path, c)

                    if not os.path.exists(task_class_folder_path):
                        os.makedirs(task_class_folder_path)

                    all_imgs = os.listdir(pths)

                    samples = all_imgs[0:num_imgs_per_cat]

                    for sp in samples:
                        shutil.copy(os.path.join(DATASET_DIR, 'train', c, sp), task_class_folder_path)
                        imgs.append((os.path.join(DATASET_DIR, 'train', c, sp), class_to_idx[c]))

                self.data = datasets.ImageFolder(task_folder_path, self.transform)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)


class GenericDataset_mm(data.Dataset):

    def __init__(self, dir_name, dataset_name, split, task_iter_item=0, subtask_class_num=5, random_sized_crop=False,
                 num_imgs_per_cat=300, label_division=True):

        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat

        global subtask_all_id
        global time_now

        if self.dataset_name == 'cifar100':

            self.mean_pix = CIFAR100_TRAIN_MEAN
            self.std_pix = CIFAR100_TRAIN_STD

            if self.split != 'train':
                transforms_list = [
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    lambda x: np.asarray(x),
                ]

            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]

            self.transform = transforms.Compose(transforms_list)
            split_data_dir = DATASET_DIR + '/novel/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)

            global subtask_all_id
            global time_now
            if self.split == 'train':
                subtask_all_id[task_iter_item] = random.sample(range(0, 20), subtask_class_num)
                re = 'inheritable_traindata_' + str(num_imgs_per_cat)
            else:
                re = 'inheritable_testdata_50_' + str(num_imgs_per_cat)

            classes = [k for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]

            class_to_idx = {k: v for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}

            imgs = []

            task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'inheritabledataset', \
                                            'Task_' + str(task_iter_item), re)

            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)

            with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
                file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, re, class_to_idx))

            for c in classes:

                pths = os.path.join(DATASET_DIR + '/novel/' + self.split, c)

                task_class_folder_path = os.path.join(task_folder_path, c)

                if not os.path.exists(task_class_folder_path):
                    os.makedirs(task_class_folder_path)

                all_imgs = os.listdir(pths)

                if self.split == 'train':
                    num_sample = min(len(all_imgs), num_imgs_per_cat)
                    samples = random.sample(all_imgs[400:], num_sample)

                else:
                    num_sample = len(all_imgs)
                    samples = random.sample(all_imgs, num_sample)

                for sp in samples:
                    shutil.copy(os.path.join(DATASET_DIR + '/novel/' + self.split, c, sp),
                                task_class_folder_path)
                    imgs.append((os.path.join(DATASET_DIR + '/novel/' + self.split, c, sp), class_to_idx[c]))

            self.data = datasets.ImageFolder(task_folder_path, self.transform)

        if self.dataset_name == 'imagenet100':

            self.mean_pix = IMAGENET100_TRAIN_MEAN
            self.std_pix = IMAGENET100_TRAIN_STD

            if self.split != 'train_mm':
                transforms_list = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x),
                ]

            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]

            self.transform = transforms.Compose(transforms_list)

            if self.split == 'train_mm':
                split_data_dir = DATASET_DIR + '/train'
            else:
                split_data_dir = DATASET_DIR + '/val'

            self.data = datasets.ImageFolder(split_data_dir, self.transform)

            if self.split == 'train_mm':
                subtask_all_id[task_iter_item] = random.sample(range(80, 100), subtask_class_num)
                re = 'inheritable_traindata_' + str(num_imgs_per_cat)
            else:
                re = 'inheritable_testdata_50_' + str(num_imgs_per_cat)

            classes = [k for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]

            class_to_idx = {k: v for (k, v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}

            imgs = []

            task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'inheritabledataset', \
                                            'Task_' + str(task_iter_item), re)

            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)

            with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
                file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, re, class_to_idx))

            for c in classes:

                if self.split == 'train_mm':
                    pths = os.path.join(DATASET_DIR, 'train', c)
                else:
                    pths = os.path.join(DATASET_DIR, 'val', c)

                task_class_folder_path = os.path.join(task_folder_path, c)

                if not os.path.exists(task_class_folder_path):
                    os.makedirs(task_class_folder_path)

                all_imgs = os.listdir(pths)

                if self.split == 'train_mm':
                    num_sample = min(len(all_imgs), num_imgs_per_cat)

                    samples = random.sample(all_imgs[400:], num_sample)

                    for sp in samples:
                        shutil.copy(os.path.join(DATASET_DIR, 'train', c, sp), task_class_folder_path)
                        imgs.append((os.path.join(DATASET_DIR, 'train', c, sp), class_to_idx[c]))

                else:
                    num_sample = len(all_imgs)

                    samples = random.sample(all_imgs, num_sample)
                    for sp in samples:
                        shutil.copy(os.path.join(DATASET_DIR, 'val', c, sp), task_class_folder_path)
                        imgs.append((os.path.join(DATASET_DIR, 'val', c, sp), class_to_idx[c]))

            self.data = datasets.ImageFolder(task_folder_path, self.transform)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

def get_cifar100_dataloaders(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cat_train, num_imgs_per_cat_test,
                             label_division):
    dataloader_train = {}
    dataloader_test = {}

    dataloader_train_mm = None

    dir_name = 'data_cifar100'

    # continual data
    for i in range(num_tasks):
        dataset_train = GenericDataset_ll(dir_name, 'cifar100', 'base', task_iter_item=i,
                                          subtask_class_num=subtask_classes_num,
                                          random_sized_crop=False, num_imgs_per_cat=num_imgs_per_cat_train,
                                          label_division=label_division)
        print('lifelong data Task {0} done! '.format(i))

    inh = [10, 20]

    # target data
    for i in range(0, num_tasks):

        for mm_vol in inh:
            dataset_train_mm = GenericDataset_mm(dir_name, 'cifar100', 'train', task_iter_item=i,
                                                 subtask_class_num=subtask_classes_num,
                                                 random_sized_crop=False, num_imgs_per_cat=mm_vol, label_division=True)

            dataset_test = GenericDataset_mm(dir_name, 'cifar100', 'test', task_iter_item=i,
                                             subtask_class_num=subtask_classes_num,
                                             random_sized_crop=False, num_imgs_per_cat=mm_vol, label_division=True)

        print('inheritable data Task {0} done! '.format(i))

    return dataset_train, dataloader_train_mm, dataloader_test


def get_permute_imagenet(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cate):
    dataloader_train_ll = {}
    dataloader_train_mm = {}
    dataloader_test = {}

    dir_name = 'data_imagenet'

    for i in range(num_tasks):
        dataset_train_ll = GenericDataset_ll(dir_name, 'imagenet', 'train_ll', task_iter_item=i,
                                             subtask_class_num=subtask_classes_num,
                                             random_sized_crop=False, num_imgs_per_cat=400, label_division=True)

        print('lifelong data Task {0} done! '.format(i))

    inh = [10, 20]

    for i in range(0, num_tasks):

        for mm_vol in inh:
            dataset_train_mm = GenericDataset_mm(dir_name, 'imagenet', 'train_mm', task_iter_item=i,
                                                 subtask_class_num=subtask_classes_num,
                                                 random_sized_crop=False, num_imgs_per_cat=mm_vol, label_division=True)

            dataset_test = GenericDataset_mm(dir_name, 'imagenet', 'val', task_iter_item=i,
                                             subtask_class_num=subtask_classes_num,
                                             random_sized_crop=False, num_imgs_per_cat=mm_vol, label_division=True)

        print('inheritable data Task {0} done! '.format(i))

    return dataloader_train_ll, dataloader_train_mm, dataloader_test



if __name__ == '__main__':
    args = parser.parse_args()

    if args.data_name == 'cifar100':
        num_tasks = 50

        num_epochs = 1

        batch_size = 64

        train_ll_loader, train_mm_loader, test_loader = get_cifar100_dataloaders(num_tasks, batch_size,
                                                                                 subtask_classes_num=5,
                                                                                 num_imgs_per_cat_train=args.num_imgs_per_cat_train,
                                                                                 num_imgs_per_cat_test=100,
                                                                                 label_division=True)

    if args.data_name == 'imagenet100':
        num_tasks = 50

        num_epochs = 1

        batch_size = 16

        train_ll_loader, train_mm_loader, test_loader = get_permute_imagenet(num_tasks, batch_size,
                                                                             subtask_classes_num=5,
                                                                             num_imgs_per_cat_train=args.num_imgs_per_cat_train,
                                                                             num_imgs_per_cate=200)
