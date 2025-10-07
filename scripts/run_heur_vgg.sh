#!/bin/bash
# 该脚本演示了如何运行 heur_vgg 算法的完整流程。

echo "========================================================"
echo "Starting Learngene Experiment: heur_vgg"
echo "========================================================"

# --- 步骤 1: 训练祖先模型 ---
echo "Step 1: Training Ancestor Model..."
# 定义配置文件路径
ANCESTOR_CONFIG="configs/ancestor_heur_vgg.yaml"

# 检查配置文件是否存在
if [ ! -f "$ANCESTOR_CONFIG" ]; then
    echo "Error: Ancestor config file not found at $ANCESTOR_CONFIG"
    exit 1
fi

# 执行训练命令
./scripts/learngene train --config $ANCESTOR_CONFIG

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
    echo "Error: Ancestor model training failed."
    exit 1
fi
echo "Ancestor model training completed successfully."

# --- 步骤 2: 提取基因并适配后代模型 ---
echo ""
echo "Step 2: Adapting Descendant Model..."
# 定义配置文件路径
DESCENDANT_CONFIG="configs/descendant_heur_vgg.yaml"

# 检查配置文件是否存在
if [ ! -f "$DESCENDANT_CONFIG" ]; then
    echo "Error: Descendant config file not found at $DESCENDANT_CONFIG"
    exit 1
fi

# 执行适配命令
./scripts/learngene adapt --config $DESCENDANT_CONFIG

if [ $? -ne 0 ]; then
    echo "Error: Descendant model adaptation failed."
    exit 1
fi
echo "Descendant model adaptation completed successfully."

echo "========================================================"
echo "Experiment heur_vgg finished."
echo "========================================================"