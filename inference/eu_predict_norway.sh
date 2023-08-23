#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python predict_labels.py \
--task_name eu_related \
--country norway \
--model_name_or_path ./eu_related/checkpoints \
--max_seq_length 512 \
--pad_to_max_length False \
--train_file ./eu_related/datasets/train.csv \
--per_device_eval_batch_size 128 \
--test_file ./data/norway_corpus_clean.csv \
--do_predict \
--dataloader_num_workers 10 \
--dataloader_pin_memory True \
--output_dir ./eu_related/checkpoints
