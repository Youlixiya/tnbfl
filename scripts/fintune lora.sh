export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python -m torch.distributed.run --master_port=24999 --nproc_per_node=1 \
         tinyllama_ft/train/finetune_mem.py \
        --lora_enable True --lora_r 8 --lora_alpha 32 \
        --data_path data/train_dev_11k.json \
        --model_name_or_path ckpts/TinyLlama-1.1B-Chat-v1.0 \
        --bf16 True \
        --output_dir checkpoints/tinyllama_ft \
        --max_steps 1000    \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 2000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --tf32 True  \
        --model_max_length 2048  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \