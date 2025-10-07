import argparse
from yacs.config import CfgNode as CN

def get_default_config():
    """
    提供一个基础的配置结构，避免在每个YAML中重复定义。
    """
    cfg = CN()

    cfg.METHOD_FAMILY = '' # 'heur_vgg' or 'tleg_vit'
    cfg.OUTPUT_DIR = './outputs/default_output'
    cfg.SEED = 42
    cfg.DEVICE = 'cuda'
    
    # 数据集配置
    cfg.DATASET = CN()
    cfg.DATASET.NAME = 'cifar100'
    cfg.DATASET.PATH = '/path/to/dataset'
    
    # 模型配置
    cfg.MODEL = CN()
    cfg.MODEL.TYPE = 'netwider'
    cfg.MODEL.PRETRAINED = False
    
    # 训练配置
    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 100
    cfg.TRAIN.BATCH_SIZE = 64
    cfg.TRAIN.LR = 0.01

    # 分布式训练 (主要用于 TLEG)
    cfg.DISTRIBUTED = CN()
    cfg.DISTRIBUTED.WORLD_SIZE = 1
    cfg.DISTRIBUTED.DIST_URL = 'env://'

    return cfg


def setup_config(config_file):
    """
    加载默认配置，并用指定的YAML文件内容覆盖它。
    """
    cfg = get_default_config()
    if config_file:
        cfg.merge_from_file(config_file)
    
    # 为了兼容 TLEG 的 argparse 风格
    # 将 CfgNode 转换为类似 argparse.Namespace 的对象
    class CfgNamespace:
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                   setattr(self, a, [CfgNamespace(x) if isinstance(x, dict) else x for x in b])
                else:
                   setattr(self, a, CfgNamespace(b) if isinstance(b, dict) else b)

    return CfgNamespace(cfg.dump())