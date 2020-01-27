#!/bin/bash
#SBATCH --job-name=finetune_bert_enron
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

python ../run_lm_finetuning.py  \
    --output_dir='output/' \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file='../enron_lm.tsv' \
    --mlm \
    --overwrite_cache \
    --overwrite_output_dir \
    --logging_steps=100 \
    --per_gpu_train_batch_size=128
