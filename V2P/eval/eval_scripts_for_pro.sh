#!/bin/bash

# 模型路径目录
training_type=$1  # warmup or sft
model_dir="models/V2P-${training_type}"

# 保存路径和数据路径
save_base_path="eval/eval_results/screenspot_pro_${training_type}"
data_path="eval/eval_datasets/ScreenSpot-Pro"

# 遍历模型目录中的所有以 checkpoint 开头的文件夹
for model in "$model_dir"/checkpoint*; do
    # 获取文件或文件夹的名字
    model_name=$(basename "$model")

    # 构造对应的保存路径
    save_path="$save_base_path/${model_name}"

    if [ -d "$save_path" ]; then
        echo "Skipping model: $model_name, results already exist at: $save_path"
        continue
    fi
    
    # 打印当前运行信息（可选）
    echo "Evaluating model: $model_name"
    echo "Saving results to: $save_path"

    # 运行评估命令
    python eval/screenSpot_pro.py --save_path "$save_path" --data_path "$data_path" --model_name_or_path "$model"
done