#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#. ./path.sh || exit 1;
set -e           # 任何命令失败时立即退出
set -o pipefail  # 管道中任何命令失败时整个管道失败
set -u           # 使用未定义变量时报错
stage=3
stop_stage=7

pretrained_model_dir=/home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B
# 在脚本开头添加
finetune_model_dir=/home/chihan/workspace/CosyVoice/pretrained_models/jianghuai
mkdir -p $finetune_model_dir
# train llm
export CUDA_VISIBLE_DEVICES="2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  
  # 修改数据路径，使用东北方言数据而非LibriSpeech数据
  cp /home/chihan/workspace/CosyVoice/data/Jiang-Huai_train/parquet/data.list data/train.data.list
  cp /home/chihan/workspace/CosyVoice/data/Jiang-Huai_val/parquet/data.list data/dev.data.list
  
  # NOTE will update llm/hift training later
  for model in llm flow; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config /home/chihan/workspace/CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path /home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config /home/chihan/workspace/CosyVoice/examples/libritts/cosyvoice2/conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi
# 复制必要文件
cp -r $pretrained_model_dir/* $finetune_model_dir/
cp `pwd`/exp/cosyvoice2/llm/torch_ddp/llm.pt $finetune_model_dir/
cp `pwd`/exp/cosyvoice2/flow/torch_ddp/flow.pt $finetune_model_dir/

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $finetune_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $finetune_model_dir
fi