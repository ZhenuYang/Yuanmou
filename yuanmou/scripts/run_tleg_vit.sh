#!/bin/bash
# 该脚本演示了如何运行 tleg_vit 算法的完整流程。

echo "========================================================"
echo "Starting Learngene Experiment: tleg_vit"
echo "========================================================"

# --- 步骤 1: 训练祖先模型/基因 ---
echo "Step 1: Training Ancestor/Gene Model (Aux-Net)..."
ANCESTOR_CONFIG="configs/ancestor_tleg_vit.yaml"

if [ ! -f "$ANCESTOR_CONFIG" ]; then
    echo "Error: Ancestor config file not found at $ANCESTOR_CONFIG"
    exit 1
fi

./scripts/learngene train --config $ANCESTOR_CONFIG

if [ $? -ne 0 ]; then
    echo "Error: Ancestor model training failed."
    exit 1
fi
echo "Ancestor/Gene model training completed successfully."

# --- 步骤 2: 适配后代模型 ---
echo ""
echo "Step 2: Adapting Descendant Model (Des-Net)..."
DESCENDANT_CONFIG="configs/descendant_tleg_vit.yaml"

if [ ! -f "$DESCENDANT_CONFIG" ]; then
    echo "Error: Descendant config file not found at $DESCENDANT_CONFIG"
    exit 1
fi

./scripts/learngene adapt --config $DESCENDANT_CONFIG

if [ $? -ne 0 ]; then
    echo "Error: Descendant model adaptation failed."
    exit 1
fi
echo "Descendant model adaptation completed successfully."

echo "========================================================"
echo "Experiment tleg_vit finished."
echo "========================================================"