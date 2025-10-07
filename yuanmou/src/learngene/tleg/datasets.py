# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import os.path
import sys
import json

from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform



class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class Food101(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Food101, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split 

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.list = []

        path_json = os.path.join(root, 'split.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        if split == 'train':
            for elem in data['train']:
                self.list.append((elem[0], elem[1]))
        else:
            for elem in data['test']:
                self.list.append((elem[0], elem[1]))

    def __getitem__(self, index):
        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        dir = self.root + "/images/" + self.list[index][0]

        image = pil_loader(dir)
        label = self.list[index][1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # Provide a way to get the length (number of elements) of the dataset
        length = len(self.list)
        return length


class miniImagenet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "test")

        if (self.Train):
            self.train_transform = transforms.Compose([
                transforms.CenterCrop(224)
            ])
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_test()
        
        
        self._make_dataset(Train = self.Train)
        

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    
    def _create_class_idx_dict_test(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.test_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.test_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

        
    def _make_dataset(self, Train=True):

        self.data = []
        self.targets = []
        
            
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.test_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            
            if not os.path.isdir(dirs):
                continue
            
            for root, _, files in sorted(os.walk(dirs)):
                
                for fname in sorted(files):
                    if (fname.endswith(".jpg")):
                        path = os.path.join(root, fname)
                        
                        if Train:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                       
                        sample = Image.open(path)
                        sample = sample.convert('RGB')
                        
                        if Train:
                            sample = self.train_transform(sample)

                        self.data.append(sample)
                        self.targets.append(class_index)
                        
                
    def __len__(self):
        return self.len_dataset

    
    def __getitem__(self, idx):
        sample, tgt = self.data[idx], self.targets[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
    

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        
        self.data = []
        self.targets = []
        
            
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            class_index = self.class_to_tgt_idx[self.val_img_to_class[fname]]
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                       
                        sample = Image.open(path)
                        sample = sample.convert('RGB')
                        self.data.append(sample)
                        self.targets.append(class_index)
                        

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sample, tgt = self.data[idx], self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
    
    

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    
    elif args.data_set == 'MiniIMNET':
        dataset = miniImagenet(root=args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    
    elif args.data_set == 'TinyIMNET':
        dataset = TinyImageNet(root=args.data_path, train=is_train, transform=transform)
        nb_classes = 200
        
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
        
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'food101':
        dataset = Food101(root=args.data_path, split='train' if is_train else 'test', transform=transform)
        nb_classes = 101

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
