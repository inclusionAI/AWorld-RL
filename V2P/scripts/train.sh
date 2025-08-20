#!/bin/bash
# model_type: qwen2vl or qwen25vl
model_type="qwen25vl"
llm_model="models/V2P-warmup"
output_dir="models/V2P-sft"

# === Training Command ===
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${RANK} --master_port=${MASTER_PORT} --master_addr=${MASTER_ADDR} train.py \
  --deepspeed scripts/zero3.json \
  --data_path data/data_config_cleaned.yaml \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels 5720064 \
  --unfreeze_all_parameters True \
  --unfreeze_pointer_head False \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_new_tokens False \
  --unfreeze_visual False \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 1.0 \
  --label_style "gauss" \
  --sigma_factor 1