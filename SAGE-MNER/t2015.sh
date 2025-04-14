DATASET_NAME="twitter15"
BERT_NAME="xlm-roberta-large"


CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --dataset_choose=${DATASET_CHOOSE} \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=32 \
        --batch_size=16 \
        --bert_lr=2e-5 \
        --crf_lr=3e-1 \
        --other_lr=1e-3 \
        --warmup_ratio=0.1 \
        --eval_begin_epoch=3 \
        --seed=42 \
        --do_train \
        --ignore_idx=0 \
        --save_path=your_ckpt_path \
        --dropout_prob=0.5 \
        --negative_slope1=0.01 \
        --negative_slope2=0.01 \
        --dyn_k=10 \
        --embed_dim=1024
        
        
