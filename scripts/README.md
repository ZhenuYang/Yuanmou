
# 脚本使用说明

该目录包含项目的主要命令行工具 `learngene` 和一系列用于自动化实验流程的 Shell 脚本模板。

## 主要工具: `learngene`

这是与本框架交互的唯一入口。所有功能都通过其子命令实现。

- **`./learngene train --config <path>`**: 训练祖先模型。
- **`./learngene adapt --config <path>`**: 适配后代模型。

## 实验脚本模板

我们提供以下 `.sh` 脚本作为示例，您可以复制并修改它们来运行您自己的实验。

- **`run_heur_vgg.sh`**: 运行 `heur_vgg` 算法的完整流程。
- **`run_tleg_vit.sh`**: 运行 `tleg_vit` 算法的完整流程。
- **`train_all_ancestors.sh`**: 一键训练所有算法的祖先模型。
- **`adapt_all_descendants.sh`**: 一键适配所有算法的后代模型。
- **`sweep_tleg_depth.sh`**: (高级示例) 自动测试 `tleg_vit` 算法生成不同深度后代模型的效果。

**使用方法**:
1.  首先赋予脚本执行权限: `chmod +x *.sh`
2.  打开脚本，修改内部的配置文件路径或参数。

3.  直接运行脚本，例如: `./run_tleg_vit.sh`
