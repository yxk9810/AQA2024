#!/bin/bash
#训练gte-qwen-7b-instruct
embedding_dir=gte_embeddings
model_path=/mnt/workspace/data/AIME/index/wjd/tevatron/sent_embedding/sentenc_models/gte_Qwen1.5-7B-instruct
data_path=/mnt/workspace/data/AIME/index/wjd/tevatron/aqa_train_data_processed/train_qwen_0507.jsonl
corpus_data=/mnt/workspace/data/AIME/index/wjd/tevatron/aqa_train_data_processed/test_corpus_data.jsonl
query_data=/mnt/workspace/data/AIME/index/wjd/tevatron/aqa_train_data_processed/test_data_0606.tsv
output_dir=retriever-gte-7b

deepspeed --include localhost:0,1 --master_port 52000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir $output_dir \
  --dataset_path $data_path \
  --model_name_or_path $model_path \
  --lora \
  --lora_target_modules q_proj,v_proj \
  --save_steps 500 \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query \nQuery:" \
  --passage_prefix "" \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --report_to none \
  --temperature 0.01 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --train_group_size 1 \
  --learning_rate 1e-4 \
  --query_max_len 64 \
  --passage_max_len 256 \
  --num_train_epochs 4 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4

#创建目录
mkdir -p $embedding_dir
#编码passage
for s in 0 1
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $output_dir \
  --lora \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query \nQuery:" \
  --passage_prefix "" \
  --fp16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 32 \
  --query_max_len 64 \
  --passage_max_len 256 \
  --dataset_path $corpus_data \
  --dataset_number_of_shards 2 \
  --dataset_shard_index ${s} \
  --encode_output_path $embedding_dir/corpus.${s}.pkl
done

#query 编码
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --lora_name_or_path $output_dir \
  --lora \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query \nQuery:" \
  --passage_prefix "" \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 16 \
  --query_max_len 512 \
  --dataset_path $query_data \
  --encode_output_path $embedding_dir/query-test.pkl


#检索输出top50
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $embedding_dir/query-test.pkl \
    --passage_reps $embedding_dir/corpus*.pkl \
    --depth 50 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to results/gte_7b_top50.txt
