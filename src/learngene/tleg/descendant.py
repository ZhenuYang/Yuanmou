from .main import main

def adapt(cfg):
    """
    适配 TLEG 后代模型 (Des-Net) 的入口函数。
    它直接调用 tleg/main.py 中的核心训练逻辑。
    
    Args:
        cfg (Namespace): 从 YAML 文件加载并转换后的配置对象。
    """
    # 适配后代模型时，必须指定学习基因的来源。
    if not hasattr(cfg.MODEL, 'LOAD_GENE') or not cfg.MODEL.LOAD_GENE:
        raise ValueError("错误: 适配后代模型时，配置文件中必须指定 'MODEL.LOAD_GENE' 路径。")
    
    main(cfg)