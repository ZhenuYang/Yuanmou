#!/bin/bash
# 该脚本用于一键训练所有已定义的祖先模型。

echo "========================================================"
echo "Starting Training for ALL Ancestor Models"
echo "========================================================"

# 定义所有祖先模型的配置文件列表
ANCESTOR_CONFIGS=(
    "configs/ancestor_heur_vgg.yaml"
    "configs/ancestor_tleg_vit.yaml"
)

# 循环执行训练
for config in "${ANCESTOR_CONFIGS[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Training with config: $config"
    echo "--------------------------------------------------------"
    
    if [ ! -f "$config" ]; then
        echo "Warning: Config file not found, skipping: $config"
        continue
    fi
    
    ./scripts/learngene train --config "$config"
    
    if [ $? -ne 0 ]; then
        echo "Error during training with $config. Aborting."
        exit 1
    fi
done

echo "========================================================"
echo "All ancestor models have been trained successfully."
echo "========================================================"