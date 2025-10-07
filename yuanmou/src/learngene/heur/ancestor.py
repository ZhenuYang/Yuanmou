import time
import datetime
import os
import copy
import xlwt

import torch
import torch.nn as nn
import torch.optim as optim

from . import utils
from . import models

def train(cfg):
    """
    训练祖先模型 (封装自 train_learngene.py 的核心逻辑)。
    这个过程是多任务的持续学习，逐步扩展网络以适应新任务。
    """
    print("开始训练祖先模型 (Netwider)...")
    
    # --- 1. 初始化和设置 ---
    utils.set_seed(cfg.SEED)
    cuda_enabled = cfg.DEVICE == 'cuda' and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.set_device(0) # 假设使用第一个GPU

    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    val_result_path = os.path.join(cfg.OUTPUT_DIR, 'validation_log.txt')
    print(f"详细日志将保存在: {val_result_path}")

    # --- 2. 加载数据 ---
    print("加载数据集...")
    with open(val_result_path, 'a') as f:
        f.write('Data loading...\n\n')

    train_loader = utils.get_permute(
        data_name=cfg.DATASET.NAME,
        num_works=cfg.DATASET.NUM_WORKS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        path=cfg.DATASET.PATH
    )

    # --- 3. 初始化模型 ---
    print("初始化模型...")
    with open(val_result_path, 'a') as f:
        f.write('Model loading...\n\n')

    model = models.Netwider(cfg.MODEL.BASE_LAYERS)
    if cuda_enabled:
        model = model.cuda()
    
    # 创建Excel日志记录器
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    # --- 4. 多任务持续学习循环 ---
    for task in range(cfg.DATASET.NUM_WORKS):
        print(f"\n========== Task {task} begins ==========")
        with open(val_result_path, 'a') as f:
            f.write(f'Task {task}:\n')

        # 从第二个任务开始，扩展网络
        if task > 0:
            model_ = copy.deepcopy(model)
            del model
            model = model_
            model.wider(task - 1)
            if cuda_enabled:
                model = model.cuda()

        print("当前模型结构:")
        model.printf()
        with open(val_result_path, 'a') as f:
            f.write(f'Current Model Structure...\n{model}\n\n')

        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        
        # 定义当前任务的模型检查点保存路径
        task_snapshot_dir = os.path.join(cfg.OUTPUT_DIR, f'task_{task}')
        os.makedirs(task_snapshot_dir, exist_ok=True)
        
        sheet_task = book.add_sheet(f'Task_{task}', cell_overwrite_ok=True)
        sheet_task.write(0, 0, 'Epoch')
        sheet_task.write(0, 1, 'Train Loss')
        sheet_task.write(0, 2, 'Train Acc')

        # --- 5. 单个任务的训练循环 ---
        for epoch in range(cfg.TRAIN.EPOCHS):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch + 1}/{cfg.TRAIN.EPOCHS} ---")
            with open(val_result_path, 'a') as f:
                f.write(f'Epoch {epoch + 1}:\n')
            
            # 训练一个epoch
            train_loss, train_acc = utils.train_epoch(
                train_loader[task](epoch), 
                model, 
                criterion, 
                optimizer, 
                cuda_enabled
            )

            print(f"Training: Average loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            with open(val_result_path, 'a') as f:
                f.write(f'Training: Average loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}\n')

            # 记录到Excel
            sheet_task.write(epoch + 1, 0, epoch + 1)
            sheet_task.write(epoch + 1, 1, train_loss)
            sheet_task.write(epoch + 1, 2, train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            
            # 保存检查点
            is_best = train_acc > best_acc
            best_acc = max(train_acc, best_acc)
            
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, output_dir=task_snapshot_dir)

            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1} training time: {epoch_end_time - epoch_start_time:.2f}s")
            
        print(f"========== Task {task} finished ==========")

    # --- 6. 保存最终结果 ---
    excel_path = os.path.join(cfg.OUTPUT_DIR, f'{cfg.DATASET.NAME}_ancestor_training_log.xls')
    book.save(excel_path)
    print(f"\n祖先模型所有任务训练完成。")
    print(f"最终模型权重保存在各个 task 子目录中，最后一个任务的模型在: {os.path.join(cfg.OUTPUT_DIR, f'task_{cfg.DATASET.NUM_WORKS-1}')}")
    print(f"详细训练日志保存在: {excel_path}")