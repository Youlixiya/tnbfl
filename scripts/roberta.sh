export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=offline

python -m torch.distributed.run --nproc_per_node=1 \
         roberta_train.py \
        --data_path data/train_dev.txt \
        --model_name_or_path ckpts/chinese-roberta-wwm-ext-large \
        --fp16 \
        --output_dir checkpoints/roberta \
        --max_steps 5000    \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 10000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --model_max_length 256  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \
        --pretraining_length 256