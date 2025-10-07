#!/bin/bash
# 这是一个高级示例脚本，用于自动化测试 TLEG 算法生成不同深度后代模型的效果。
# 它会临时修改配置文件，运行实验，然后恢复配置文件。

echo "========================================================"
echo "Starting Hyperparameter Sweep for TLEG Descendant Depth"
echo "========================================================"

# 基础配置文件
BASE_CONFIG="configs/descendant_tleg_vit.yaml"
# 临时配置文件的备份名
BACKUP_CONFIG="${BASE_CONFIG}.bak"

# 检查基础配置文件是否存在
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config file not found at $BASE_CONFIG"
    exit 1
fi

# 定义要测试的后代模型深度和类型
declare -A DEPTH_MAP
DEPTH_MAP=(
    [3]="deit_tiny_patch16_224_L3"
    [6]="deit_tiny_patch16_224_L6"
    [9]="deit_tiny_patch16_224_L9"
    [12]="deit_tiny_patch16_224_L12"
)

# 备份原始配置文件
cp "$BASE_CONFIG" "$BACKUP_CONFIG"
echo "Backed up original config to $BACKUP_CONFIG"

# 循环测试不同深度
for depth in "${!DEPTH_MAP[@]}"; do
    model_type="${DEPTH_MAP[$depth]}"
    output_dir="./outputs/tleg_vit/sweep/descendant_L${depth}"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Testing Depth: ${depth}, Model: ${model_type}"
    echo "Output will be in: ${output_dir}"
    echo "--------------------------------------------------------"

    # 使用 sed 命令动态修改配置文件中的模型类型和输出目录
    sed -i.tmp "s|TYPE:.*|TYPE: '${model_type}'|g" "$BASE_CONFIG"
    sed -i.tmp "s|OUTPUT_DIR:.*|OUTPUT_DIR: '${output_dir}'|g" "$BASE_CONFIG"
    
    # 运行适配命令
    ./scripts/learngene adapt --config "$BASE_CONFIG"
    
    if [ $? -ne 0 ]; then
        echo "Error during adaptation for depth ${depth}. Restoring config and aborting."
        # 出错时也要恢复配置文件
        mv "$BACKUP_CONFIG" "$BASE_CONFIG"
        exit 1
    fi
done

# 恢复原始配置文件
mv "$BACKUP_CONFIG" "$BASE_CONFIG"
rm -f "${BASE_CONFIG}.tmp"
echo ""
echo "Sweep finished. Original config file has been restored."
echo "========================================================"