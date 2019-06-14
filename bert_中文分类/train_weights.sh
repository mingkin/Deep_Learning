#!/bin/sh

export BERT_BASE_DIR=/data/antispam/kingming/bert-master/chinese_L-12_H-768_A-12
export GLUE_DIR=/data/antispam/kingming/bert-master

python run_classifier_weights.py \
  --task_name=ming \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/tt \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --weights='1,10' \
  --output_dir=/data/antispam/kingming/bert-master/output/
