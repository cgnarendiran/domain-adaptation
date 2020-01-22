#!/bin/bash
#SBATCH --job-name=finetune_bert_enron
#SBATCH --partition=short
#SBATCH --time=5:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgnarendiran@gmail.com

python ../run_lm_finetuning.py  \
    --output_dir='output/' \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file='../enron_lm.csv' \
    --mlm
