from .main import main

def train(cfg):
    """
    训练 TLEG 祖先模型 (Aux-Net) 的入口函数。
    它直接调用 tleg/main.py 中的核心训练逻辑。
    
    Args:
        cfg (Namespace): 从 YAML 文件加载并转换后的配置对象。
    """
    # 对于祖先模型训练，我们确保没有设置 LOAD_GENE 路径，
    # 因为它是从头开始或从 ImageNet 预训练模型开始训练的。
    if hasattr(cfg.MODEL, 'LOAD_GENE') and cfg.MODEL.LOAD_GENE:
        print("警告: 正在训练祖先模型，但配置文件中设置了 'MODEL.LOAD_GENE'。该设置将被忽略。")
        # 创建一个副本以避免修改原始cfg对象
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.MODEL.LOAD_GENE = ''
        main(cfg_copy)
    else:
        main(cfg)