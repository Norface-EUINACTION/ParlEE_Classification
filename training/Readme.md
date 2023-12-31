# Training script for EU and CAP classifiers.

To run the training for EU and CAP, simple run `./eu_train_eval.sh` and `./cap_train_eval.sh`. **(TODO: ADD TRAIN AND TEST SPLITS FOR EU AND CAP)** You can find the training and validation data for EU and CAP in the dataset folders

Both the scripts are similar. For example, the below script is for training a CAP classifier. To change the training from CAP to EU, simply replace `--task_name` with *eu_related*, the `--train_file`. The training and eval data are in the datasets folder for eu and cap. 
and `--validation_file` paths to the paths of EU training and validation set, and `--output_dir`. 

```
#!/bin/bash
python text_classification.py \
--task_name cap \
--model_name_or_path  xlm-roberta-base \
--max_seq_length 512 \
--pad_to_max_length False \
--fp16 True \
--train_file ./datasets/cap/train.csv \
--validation_file ./datasets/cap/eval.csv \
--do_train \
--do_eval \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 32 \
--dataloader_num_workers 10 \
--dataloader_pin_memory True \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--output_dir ./cap/checkpoints \
--save_total_limit 1 \
--evaluation_strategy epoch \
--load_best_model_at_end True \
--metric_for_best_model eval_f1
```
