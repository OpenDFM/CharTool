export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --freeze_vit True \
    --dataset 'data/tool_sft.jsonl' \
    --load_from_cache_file true \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --max_length 32768 \
    --save_only_model true \
    --output_dir /your_ckpts/chartool-3b-sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --use_chat_template true \
    --deepspeed zero3 \
    --padding_free true \
    --attn_impl flash_attn
