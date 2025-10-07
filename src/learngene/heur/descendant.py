import os
import copy
import xlwt
import torch
import torch.nn as nn
import torch.optim as optim

from . import utils
from . import models
from .extraction import extract

def initialize_and_adapt(cfg):
    """
    封装了 initialize_w_learngene.py 的核心逻辑。
    首先提取学习基因，然后用它初始化后代模型，并进行适配训练。
    """
    print("========================================================")
    print("开始初始化并适配后代模型 (Descendant Model)")
    print("========================================================")

    # --- 1. 初始化和设置 ---
    utils.set_seed(cfg.SEED)
    cuda_enabled = cfg.DEVICE == 'cuda' and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.set_device(0)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 创建Excel日志记录器
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)

    # --- 2. 提取学习基因 ---
    # 调用 extraction 模块来获取基因
    # 注意: descendant 的配置文件需要包含 ancestor 训练时的 NUM_WORKS 信息
    learngene_layers = extract(cfg, cuda_enabled)

    # --- 3. 加载目标任务（可继承任务）的数据 ---
    print("\n加载下游任务数据集...")
    train_loaders, test_loaders = utils.get_inheritable_heur(
        data_name=cfg.DATASET.NAME,
        num_works_tt=cfg.DATASET.NUM_WORKS_TT,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_imgs_per_cate=cfg.DATASET.NUM_IMGS_PER_CAT_TRAIN,
        path=cfg.DATASET.PATH
    )
    print(f"共加载了 {len(train_loaders)} 个下游任务。")

    # --- 4. 使用基因初始化后代模型 ---
    # 创建一个基础的后代模型模板
    # linear_in_features 2048 是根据模型结构硬编码的，如果模型变化需要调整
    base_descendant_model = models.vgg_compression_ONE(
        learngene_layers, 
        linear_in_features=2048, 
        num_class=cfg.MODEL.NUM_CLASS
    )
    if cuda_enabled:
        base_descendant_model = base_descendant_model.cuda()

    print("\n已使用学习基因创建后代模型模板:")
    base_descendant_model.printf()

    # --- 5. 在每个下游任务上进行适配训练和评估 ---
    for task_tt in range(cfg.DATASET.NUM_WORKS_TT):
        print(f"\n========== 开始适配下游任务 {task_tt} ==========")
        
        # 每次都从加载了基因的原始模型深拷贝，确保每个任务的起点相同
        model_vgg = copy.deepcopy(base_descendant_model)
        
        # 为当前任务创建Excel工作表
        sheet_task = book.add_sheet(f'TT_Task_{task_tt}', cell_overwrite_ok=True)
        sheet_task.write(0, 0, 'Epoch')
        sheet_task.write(0, 1, 'Train Loss')
        sheet_task.write(0, 2, 'Train Acc')
        sheet_task.write(0, 3, 'Test Loss')
        sheet_task.write(0, 4, 'Test Acc')
        
        # --- 单个任务的训练循环 ---
        for epoch in range(cfg.TRAIN.EPOCHS):
            # 为每个epoch重新创建优化器，以重置学习率等状态（如果需要）
            optimizer = optim.SGD(
                model_vgg.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=5e-4 # 这个值在原始代码中是硬编码的
            )
            criterion = nn.CrossEntropyLoss()

            # 训练
            train_loss, train_acc = utils.train_epoch(
                train_loaders[task_tt](epoch),
                model_vgg,
                criterion,
                optimizer,
                cuda_enabled
            )
            
            # 测试
            test_loss, test_acc = utils.test_epoch(
                test_loaders[task_tt](epoch),
                model_vgg,
                criterion,
                cuda_enabled
            )
            
            print(f"Epoch {epoch + 1}/{cfg.TRAIN.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # 记录到Excel
            sheet_task.write(epoch + 1, 0, epoch + 1)
            sheet_task.write(epoch + 1, 1, train_loss)
            sheet_task.write(epoch + 1, 2, train_acc)
            sheet_task.write(epoch + 1, 3, test_loss)
            sheet_task.write(epoch + 1, 4, test_acc)

        # 保存当前任务训练好的模型 (可选)
        task_output_dir = os.path.join(cfg.OUTPUT_DIR, f'task_{task_tt}')
        os.makedirs(task_output_dir, exist_ok=True)
        torch.save(model_vgg.state_dict(), os.path.join(task_output_dir, 'final_model.pth'))
        print(f"下游任务 {task_tt} 适配完成，模型已保存。")
        del model_vgg

    # --- 6. 保存最终的Excel日志 ---
    excel_path = os.path.join(cfg.OUTPUT_DIR, f'{cfg.DATASET.NAME}_descendant_adaptation_log.xls')
    book.save(excel_path)
    
    print("\n========================================================")
    print("所有下游任务适配完成！")
    print(f"详细日志已保存在: {excel_path}")
    print("========================================================")