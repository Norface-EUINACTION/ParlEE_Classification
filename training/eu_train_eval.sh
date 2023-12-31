#!/bin/bash
python text_classification.py \
--task_name eu_related \
--model_name_or_path  xlm-roberta-base \
--max_seq_length 512 \
--pad_to_max_length False \
--fp16 True \
--train_file ./eu_related/datasets/train.csv \
--validation_file ./eu_related/datasets/eval.csv \
--do_train \
--do_eval \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 32 \
--dataloader_num_workers 10 \
--dataloader_pin_memory True \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--output_dir ./eu_related/checkpoints \
--save_total_limit 1 \
--evaluation_strategy epoch \
--load_best_model_at_end True \
--metric_for_best_model eval_f1