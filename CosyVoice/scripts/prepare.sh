#!/bin/bash
# Copyright 2024 
set -e           # 任何命令失败时立即退出
set -o pipefail  # 管道中任何命令失败时整个管道失败
set -u           # 使用未定义变量时报错
#. ./path.sh || exit 1;

stage=2
stop_stage=4

# 设置数据和模型路径
data_dir=/home/chihan/workspace/CosyVoice/data/KeSpeech_Jiang-Huai
pretrained_model_dir=pretrained_models/CosyVoice2-0.5B
output_dir=exp/Jiang-Huai_dialect

# 数据准备阶段
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "数据准备，创建Kaldi格式数据（wav.scp/text/utt2spk/spk2utt）"
  mkdir -p data/Jiang-Huai_train data/Jiang-Huai_val data/Jiang-Huai_test
  
  # 使用自定义脚本从JSONL文件生成Kaldi格式数据
  python tools/prepare_dialect_data.py \
    --jsonl_file $data_dir/dialect_train.jsonl \
    --output_dir data/Jiang-Huai_train
  
  python tools/prepare_dialect_data.py \
    --jsonl_file $data_dir/dialect_val.jsonl \
    --output_dir data/Jiang-Huai_val
  
  python tools/prepare_dialect_data.py \
    --jsonl_file $data_dir/dialect_test.jsonl \
    --output_dir data/Jiang-Huai_test
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "提取说话人嵌入，将生成spk2embedding.pt和utt2embedding.pt"
  for x in Jiang-Huai_train Jiang-Huai_val Jiang-Huai_test; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "提取语音token，将生成utt2speech_token.pt"
  for x in Jiang-Huai_train Jiang-Huai_val Jiang-Huai_test; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "准备parquet格式数据"
  for x in Jiang-Huai_train Jiang-Huai_val Jiang-Huai_test; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 100 \
      --num_processes 4 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
  
  # 创建训练和验证数据列表
  cp data/Jiang-Huai_train/parquet/data.list data/Jiang-Huai_train.list
  cp data/Jiang-Huai_val/parquet/data.list data/Jiang-Huai_val.list
fi

