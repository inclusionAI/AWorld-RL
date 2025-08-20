#!/bin/bash

# 模型路径目录
training_type=$1  # warmup or sft
model_dir="models/V2P-${training_type}"

# 保存路径基础目录
save_base_path="eval/eval_results/screenspot_v2_${training_type}"

# 模型类型
model_type="qwen25vl"

# 遍历模型目录中的所有以 checkpoint 开头的文件夹
for model in "$model_dir"/checkpoint*; do
    # 获取 checkpoint 文件夹的名字
    model_name=$(basename "$model")

    # 构造对应的保存路径
    save_path="$save_base_path/${model_name}"

    if [ -d "$save_path" ]; then
        echo "Skipping model: $model_name, results already exist at: $save_path"
        continue
    fi

    # 打印当前模型和对应的保存路径（可选，用于调试）
    echo "Evaluating model: $model_name"
    echo "Saving results to: $save_path"

    # 运行评估命令
    python eval/screenSpot_v2.py --save_path "$save_path" --model_name_or_path "$model" --model_type "$model_type"
done