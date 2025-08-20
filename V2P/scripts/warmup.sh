#!/bin/bash
# model_type: qwen2vl or qwen25vl
model_type="qwen25vl"
llm_model="models/Qwen2.5-VL-7B-Instruct"
output_dir="models/V2P-warmup"

# === Training Command ===
torchrun --nproc_per_node=1 train.py \
  --deepspeed scripts/zero3.json \
  --data_path data/data_config.yaml \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 3000 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 100000000 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels 5720064 \
  --unfreeze_all_parameters False \
  --unfreeze_pointer_head True \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_new_tokens True \
  --unfreeze_visual False \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight -1.0 \
  --label_style "gauss" \
  --sigma_factor 1