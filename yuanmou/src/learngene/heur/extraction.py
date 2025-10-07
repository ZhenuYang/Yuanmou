import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from . import models

def extract(cfg, cuda_enabled):
    """
    封装了 extract_learngene.py 的核心逻辑。
    该函数会加载一个训练到最终阶段的祖先模型，并从中提取出“学习基因”层。

    Args:
        cfg (CfgNode): 包含所需参数的配置对象。
                       需要 cfg.GENE.ANCESTOR_MODEL_PATH 和 cfg.DATASET.NUM_WORKS。
        cuda_enabled (bool): 是否使用 CUDA。

    Returns:
        nn.ModuleList: 包含被提取出的“学习基因”层的模块列表。
    """
    print("正在从祖先模型中提取学习基因...")
    
    # --- 1. 初始化一个基础的 Netwider 模型 ---
    # 这个模型将被逐步“扩展”并加载预训练权重，以重建最终的祖先模型结构
    model = models.Netwider(cfg.MODEL.BASE_LAYERS)
    if cuda_enabled:
        model = model.cuda()

    # --- 2. 模拟持续学习过程以构建最终的模型结构并加载权重 ---
    # 这个循环的目的是为了正确地重建出最后一个任务（task 20）的模型结构
    num_works = cfg.DATASET.NUM_WORKS  # 从配置中获取任务总数
    
    for task in range(num_works):
        print(f"  重建祖先模型结构: 任务 {task + 1}/{num_works}...")
        
        # 从第二个任务开始，网络结构会变宽
        if task > 0:
            model_ = copy.deepcopy(model)
            del model
            model = model_
            model.wider(task - 1)  # 这里的 layer_idx 可能是个固定值或需要配置
            if cuda_enabled:
                model = model.cuda()

        # 定义当前任务的检查点路径
        snapshot_model_dir = os.path.join(cfg.GENE.ANCESTOR_MODEL_PATH, f'task_{task}')
        checkpoint_path = os.path.join(snapshot_model_dir, 'checkpoint.pth')
        
        # 仅加载最后一个任务的权重就足够了，因为前面的权重信息已包含在内
        if task == num_works - 1:
            if os.path.isfile(checkpoint_path):
                print(f"  加载最终任务 {task} 的权重从: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if cuda_enabled else 'cpu'))
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise FileNotFoundError(f"错误: 找不到祖先模型最终任务的检查点文件: {checkpoint_path}")
    
    print("\n最终祖先模型结构重建完成:")
    model.printf()

    # --- 3. 提取预定义好的“学习基因”层 ---
    learngene_layers = model.get_layers_19_20()
    print("\n学习基因提取成功！提取的层是:")
    print(learngene_layers)
    
    return learngene_layers