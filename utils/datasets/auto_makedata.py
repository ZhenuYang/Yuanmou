from __future__ import print_function
import numpy as np
# import dill as pickle
import time
import datetime
import random
from PIL import Image
import os
import shutil
import errno
import sys
sys.path.append("../")
import csv
from pdb import set_trace as breakpoint
from matplotlib import pyplot as plt
import argparse

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

parser = argparse.ArgumentParser(description='data_mk')

parser.add_argument('--path', type=str, default='./', help='path of base classes')
parser.add_argument('--dataset', default='cifar100')

args = parser.parse_args()
DATASET_DIR = args.path #Cifar100 datapath
subtask_all_id = {}
time_now = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H_%M_%S'))

 
# ./novel______train 500 samples
#       |______test 100 samples
class GenericDataset_cifar(data.Dataset):
    
    def __init__(self, dir_name, dataset_name, split, task_iter_item = 0, subtask_class_num=5, \
                 random_sized_crop=False, num_imgs_per_cat=300):
        
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat

        
        self.mean_pix = CIFAR100_TRAIN_MEAN
        self.std_pix = CIFAR100_TRAIN_STD
        transforms_list = [
                    transforms.Resize(32), 
                    transforms.CenterCrop(32),
                    lambda x: np.asarray(x), 
                   ]
        self.transform = transforms.Compose(transforms_list)
            
        split_data_dir = DATASET_DIR + '/novel/' + self.split
        self.data = datasets.ImageFolder(split_data_dir, self.transform) 
            
        if self.split == 'train':
            subtask_all_id.update({ task_iter_item : random.sample(range(0, 20), subtask_class_num) })
            re = 'inheritable_traindata_' + str(num_imgs_per_cat)
        else:
            re = 'inheritable_testdata_' + str(num_imgs_per_cat)
            
            
        classes = [k for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]
            
        class_to_idx = {k : v for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}
            
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
                samples = random.sample(all_imgs[:500], num_sample)
                        
            else:
                num_sample = len(all_imgs) # 3
                samples = random.sample(all_imgs, num_sample)
                
            for sp in samples:
                shutil.copy(os.path.join(DATASET_DIR + '/novel/' + self.split, c, sp), task_class_folder_path)
                imgs.append((os.path.join(DATASET_DIR + '/novel/' + self.split, c, sp), class_to_idx[c]))
                
        self.data = datasets.ImageFolder(task_folder_path, self.transform)
            
            
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

#MiniImageNet
class GenericDataset(data.Dataset):
    
    def __init__(self, dir_name, dataset_name, split, task_iter_item = 0, subtask_class_num=5, \
                 random_sized_crop=False, num_imgs_per_cat=300):
        
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat

        self.mean_pix  = [0.485, 0.456, 0.406]
        self.std_pix= [0.229, 0.224, 0.225]
        transforms_list = [
                        transforms.Resize(84),
                        transforms.RandomCrop(84),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
        self.transform = transforms.Compose(transforms_list)
            
        split_data_dir = DATASET_DIR + '/test' 
        self.data = datasets.ImageFolder(split_data_dir, self.transform) 
            
        if self.split == 'train':
            subtask_all_id.update({ task_iter_item : random.sample(range(0, 20), subtask_class_num) })
            re = 'inheritable_traindata_' + str(num_imgs_per_cat)
        else:
            re = 'inheritable_testdata_' + str(num_imgs_per_cat)
            
            
        classes = [k for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]
            
        class_to_idx = {k : v for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}
            
        imgs = []
            
        task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'inheritabledataset', \
                               'Task_' + str(task_iter_item), re)
            
        if not os.path.exists(task_folder_path):
            os.makedirs(task_folder_path)

        with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
            file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, re, class_to_idx))    
                
        for c in classes:
                
            pths = os.path.join(split_data_dir, c)
                
            task_class_folder_path = os.path.join(task_folder_path, c)
                
            if not os.path.exists(task_class_folder_path):
                os.makedirs(task_class_folder_path)
                
            all_imgs = os.listdir(pths)  
                
            if self.split == 'train':
                num_sample = min(len(all_imgs), num_imgs_per_cat)
                samples = random.sample(all_imgs[:500], num_sample)
                        
            else:
                num_sample = len(all_imgs[500:]) # 3
                samples = random.sample(all_imgs[500:], num_sample)
                
            for sp in samples:
                shutil.copy(os.path.join(split_data_dir, c, sp), task_class_folder_path)
                imgs.append((os.path.join(split_data_dir, c, sp), class_to_idx[c]))
                
        self.data = datasets.ImageFolder(task_folder_path, self.transform)
            
            
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

def get_cifar100_dataloaders(num_tasks, batch_size, subtask_classes_num):
    
    dataloader_train = []
    dataloader_test = []

    dir_name = 'data_cifar100'
    
    inh = [5, 10, 20]
    #inh = [5, 10, 20, 30, 40, 50, 100]
    
    # target data
    for i in range(0, num_tasks):
        
        for mm_vol in inh:
            dataset_train = GenericDataset_cifar(dir_name, 'cifar100','train', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                       random_sized_crop=False, num_imgs_per_cat = mm_vol)
            dataloader_train.append(dataset_train)
        
            dataset_test = GenericDataset_cifar(dir_name, 'cifar100','test', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                      random_sized_crop=False, num_imgs_per_cat = mm_vol) 
            dataloader_test.append(dataset_test)
        
        print('inheritable data Task {0} done! '.format(i))
        
    return dataloader_train, dataloader_test

def get_MiniImageNet_dataloaders(num_tasks, batch_size, subtask_classes_num):
    
    dataloader_train = []
    dataloader_test = []

    dir_name = 'data_MiniImageNet'
    
    #inh = [5, 10, 20, 30, 40, 50, 100]
    inh = [5, 10, 20]
    
    
    # target data
    for i in range(0, num_tasks):
        
        for mm_vol in inh:
            dataset_train = GenericDataset(dir_name, 'MiniImageNet','train', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                       random_sized_crop=False, num_imgs_per_cat = mm_vol)
            dataloader_train.append(dataset_train)
        
            dataset_test = GenericDataset(dir_name, 'MiniImageNet','test', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                      random_sized_crop=False, num_imgs_per_cat = mm_vol) 
            dataloader_test.append(dataset_test)
        
        print('inheritable data Task {0} done! '.format(i))
        
    return dataloader_train, dataloader_test

if __name__ == '__main__':
    
    print(time_now)

    args = parser.parse_args()
    
    num_tasks = 10
    
    num_epochs = 1
    
    batch_size = 64
    
    if args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_dataloaders(num_tasks, batch_size, subtask_classes_num = 5)
        
    elif args.dataset == 'MiniImageNet':
        train_loader, test_loader = get_MiniImageNet_dataloaders(num_tasks, batch_size, subtask_classes_num = 5)