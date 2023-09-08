# Inference (Prediction) scripts for EU and CAP classifier

Models for EU and CAP are stored in the models directory. The python file *predict_labels.py* run the EU and CAP models. 

1. To run EU classifier, run './eu_predict_spain.sh'. This is just an example of one country. For other countries, you can create similar bash scripts.

2. Similarly, to run CAP classifier, run './cap_predict_spain.sh'. You can create similar bash scripts for each country for CAP classifier.

3. Both the EU and CAP bash files are same and only take difference parameters. In the below script, changing `eu_related` to `cap` will run the CAP classifier. `country` is
for which country you want to run. `model_name_or_path` requires path of the model. You can adjust `per_device_eval_batch_size` according to your GPU. The results will be stored in the `output_dir`
you specify with index as the main identifier. 

```
CUDA_VISIBLE_DEVICES=0 python predict_labels.py \
--task_name eu_related \
--country spain \
--model_name_or_path ./models/checkpoint_eu/checkpoint-21288 \
--max_seq_length 512 \
--pad_to_max_length False \
--train_file ./eu_related/datasets/train.csv \
--per_device_eval_batch_size 128 \
--test_file ./data/spain_corpus_clean.csv \
--do_predict \
--dataloader_num_workers 10 \
--dataloader_pin_memory True \
--output_dir ./models/results/eu/
```
