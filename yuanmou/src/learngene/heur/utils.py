import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

# =============================================================================
#                           General Utilities
# =============================================================================
def set_seed(seed):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    """保存模型检查点"""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    torch.save(states, os.path.join(output_dir, filename))
    
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))

# =============================================================================
#                           Data Loading Utilities
# =============================================================================
class CustomDataset(Dataset):
    """一个自定义的数据集包装器"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

def get_permute(data_name, num_works, batch_size, path):
    """为持续学习（祖先模型训练）加载数据"""
    loaders = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    for i in range(num_works):
        data_path = os.path.join(path, f'task_{i}.npz')
        data = np.load(data_path)
        train_data = torch.from_numpy(data['train_data']).float()
        train_targets = torch.from_numpy(data['train_targets']).long()
        
        dataset = CustomDataset(train_data, train_targets, transform=transform)
        
        # 使用 lambda 函数延迟创建 DataLoader
        loaders.append(lambda epoch, d=dataset: DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders

def get_inheritable_heur(data_name, num_works_tt, batch_size, num_imgs_per_cate, path):
    """为可继承任务（后代模型适配）加载数据"""
    train_loaders = []
    test_loaders = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    for i in range(num_works_tt):
        train_path = os.path.join(path, f'inheritable_train_task_{i}.npz')
        test_path = os.path.join(path, f'inheritable_test_task_{i}.npz')
        
        train_npz = np.load(train_path)
        train_data = torch.from_numpy(train_npz['data']).float()
        train_targets = torch.from_numpy(train_npz['targets']).long()
        train_dataset = CustomDataset(train_data, train_targets, transform=transform)
        train_loaders.append(lambda epoch, d=train_dataset: DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0))

        test_npz = np.load(test_path)
        test_data = torch.from_numpy(test_npz['data']).float()
        test_targets = torch.from_numpy(test_npz['targets']).long()
        test_dataset = CustomDataset(test_data, test_targets, transform=transform)
        test_loaders.append(lambda epoch, d=test_dataset: DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=0))
        
    return train_loaders, test_loaders

# =============================================================================
#                        Training and Testing Loops
# =============================================================================
def train_epoch(train_loader, model, criterion, optimizer, cuda_enabled):
    """训练一个 epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if cuda_enabled:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / (batch_idx + 1)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test_epoch(test_loader, model, criterion, cuda_enabled):
    """测试一个 epoch"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if cuda_enabled:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / (batch_idx + 1)
    accuracy = 100. * correct / total
    return avg_loss, accuracy