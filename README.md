# 集成多方言识别、情感分析与语音合成的低资源语音包

我们提供了一个功能强大的语音处理系统，集成了语音识别、说话人分离、情感识别、降噪处理和声音克隆等多种功能，为音频处理提供一站式解决方案。

## 项目背景与意义

本项目旨在构建一个能够处理多种方言、支持说话人识别与情感分析，并具备语音合成能力的智能语音处理系统，既是语音技术多维进化的体现，也对提升本地化智能服务质量、促进区域文化沟通具有重要意义。

## 功能特性

- **多方言语音识别**：支持普通话、江淮方言、东北方言、西南方言等多种方言的精准识别
- **说话人分离**：自动分离多说话人音频，识别不同说话人并生成独立音频片段
- **情感识别**：分析语音中的情感状态，支持多种情感类型（高兴、悲伤、愤怒等）
- **专业降噪**：集成UVR5降噪算法，提供多种降噪模型和可调节的降噪级别
- **声音克隆**：基于CosyVoice2的零样本声音克隆技术，仅需短音频即可模仿说话人声音
- **时间戳生成**：自动为语音内容生成精确时间戳

## 环境要求

- Python 3.10+
- CUDA 11.6+ (GPU加速，推荐使用)
- 至少8GB RAM
- 至少20GB磁盘空间（用于存储模型和处理文件）

## 安装步骤

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 下载预训练模型（如果没有自动下载）
   ```bash
   python download_models.py
   ```

3. 启动应用（在SenseVoice目录下）
   ```bash
   python app.py
   ```

## Docker部署与使用

本项目支持通过Docker容器化部署，推荐用于生产环境或快速体验。

### 1. 构建镜像

```bash
# 在项目根目录下执行
# 推荐使用国内源加速构建
# docker build --network=host -t sensevoice:latest .
docker build -t sensevoice:latest .
```

### 2. 运行容器

```bash
docker run -d --name sensevoice \
  --gpus all \
  -p 5000:5000 \
  -v /path/to/your/models/iic:/app/model_cache/models/iic \
  -v /path/to/your/CosyVoice:/app/CosyVoice \
  -v /path/to/your/tools:/app/tools \
  -v /path/to/persistent/outputs:/app/outputs \
  -v /path/to/persistent/uploads:/app/uploads \
  sensevoice:latest
```

- `/path/to/your/models/iic`：ASR和情感识别模型目录
- `/path/to/your/CosyVoice`：CosyVoice声音克隆模型目录
- `/path/to/your/tools`：降噪工具和模型目录
- `/path/to/persistent/outputs`、`/path/to/persistent/uploads`：结果和上传文件持久化目录

### 3. Dockerfile关键路径说明

Dockerfile会自动将`SenseVoice/app.py`、`SenseVoice/templates/`、`SenseVoice/static/`复制到容器内`/app/`目录下。

```dockerfile
COPY SenseVoice/app.py /app/
COPY SenseVoice/templates/ /app/templates/
COPY SenseVoice/static/ /app/static/
```

### 4. 访问服务

容器启动后，浏览器访问：
```
http://localhost:5000
```

### 5. 常见问题与建议

- 如需GPU加速，需安装NVIDIA驱动和nvidia-docker
- 可通过`--shm-size=8g`参数提升大模型推理性能
- 如需自定义模型路径或端口，可通过`-e`环境变量或`-p`参数调整
- 查看日志：`docker logs sensevoice`
- 停止/删除容器：`docker stop sensevoice && docker rm sensevoice`

## 系统架构

本系统采用模块化设计，主要包括以下核心组件：

1. **用户交互界面模块**：基于Flask的Web应用，提供直观的操作界面
2. **音频输入与预处理模块**：支持多种音视频格式，集成UVR5降噪处理
3. **说话人识别与切分模块**：基于CAM++模型，自动识别并分离不同说话人
4. **多方言语音识别模块**：基于SenseVoice微调的方言模型，实现精准识别
5. **情感识别模块**：利用SenseVoice多任务能力，分析语音情感色彩
6. **语音合成模块**：基于CosyVoice2的零样本声音克隆技术

各模块间通过统一的数据流转和API接口紧密协作，实现从音频输入到结果输出的完整处理流程。

## 使用方法

1. 启动应用后，在浏览器中访问 `http://localhost:5000`
2. 通过Web界面上传音频或视频文件
3. 选择识别模型、分段字数以及是否启用降噪
4. 点击"开始处理"按钮
5. 处理完成后，系统将显示识别结果和分离的音频片段
6. 可点击每个片段进行播放、编辑或声音克隆

### 降噪功能

1. 访问 `http://localhost:5000/denoise_page` 使用独立的降噪工具
2. 上传需要降噪的音频文件
3. 选择降噪模型和降噪级别
4. 点击"开始降噪"

### 声音克隆

1. 在识别结果页面，选择需要克隆的说话人
2. 输入要用克隆声音朗读的文本
3. 点击"生成克隆语音"
4. 系统将使用选中说话人的声音特征合成新的语音

## 模型微调

本项目支持对SenseVoice语音识别模型和CosyVoice2声音克隆模型进行微调，以适应特定方言、场景或提高性能。

### SenseVoice模型微调

SenseVoice模型微调适用于提高方言识别准确率或适应特定领域语音。本项目基于KeSpeech方言数据集完成了对江淮话、东北话和西南话三种方言的微调。

#### KeSpeech方言数据集

KeSpeech是由科大讯飞联合多家研究机构开发的大规模中文方言数据集，包含：
- **江淮方言子集**：约1,200小时录音，500+说话人
- **东北方言子集**：约1,500小时录音，600+说话人
- **西南方言子集**：约1,800小时录音，700+说话人

数据集以JSONL格式组织，包含完整的音频路径、转写文本、情感标签等信息。

#### 数据准备与预处理

1. 准备训练数据集，JSONL格式示例：
   ```json
   {"key": "1010523_1424322c", "text_language": "<|zh|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "宝宝的袜子总也找不到另一只", "source": "F:\\speech_data\\KeSpeech_Jiang-Huai\\audio\\1010523_1424322c.wav", "target_len": 13, "source_len": 5653}
   {"key": "1010633_d403ba14", "text_language": "<|zh|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "今天看片会就是来接受拍砖的", "source": "F:\\speech_data\\KeSpeech_Jiang-Huai\\audio\\1010633_d403ba14.wav", "target_len": 13, "source_len": 5878}
   ```

#### 执行微调

使用项目提供的`finetune.sh`脚本进行微调，关键参数如下：

```bash
# 指定GPU
export CUDA_VISIBLE_DEVICES="2,3"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# 基础模型（使用SenseVoiceSmall作为预训练模型）
model_name_or_model_dir="iic/SenseVoiceSmall"

# 训练和验证数据集
train_data=/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_new_augmented/0.1x_mixed_output.jsonl
val_data=/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern/dialect_val.jsonl

# 输出目录
output_dir="./outputs/aug_new_0.1x_Northeastern_finetune"
```

核心训练参数：
```bash
++train_conf.max_epoch=50 \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002
```

微调参数说明：
- `batch_size`：批处理大小，根据GPU内存调整
- `max_epoch`：最大训练轮次，通常设置为30-50
- `lr`：学习率，方言微调通常使用较小学习率(0.0001-0.0002)
- `validate_interval`：验证间隔，控制模型评估频率
- `save_checkpoint_interval`：保存检查点间隔
- `keep_nbest_models`：保留最佳模型数量，防止过拟合

### CosyVoice2模型微调

CosyVoice2模型采用端到端的多任务语音建模系统，结合了语音分块建模、语言建模与条件合成技术，支持零样本语音克隆和个性化TTS。

#### 模型架构

CosyVoice2由三个核心模块组成：
1. **监督语音分词器**：将语音编码为离散token
2. **文本-语音语言模型**：建模token序列之间的概率关系
3. **基于分块的流匹配合成模块**：利用参考音频与输入文本生成高质量语音

#### 准备数据

本项目使用了`prepare.sh`脚本进行方言数据准备，主要步骤如下：

1. **数据转换与格式准备**
   ```bash
   # 设置数据和输出路径
   data_dir=/home/chihan/workspace/CosyVoice/data/KeSpeech_Jiang-Huai
   output_dir=exp/Jiang-Huai_dialect
   
   # 从JSONL文件生成Kaldi格式数据(wav.scp/text/utt2spk/spk2utt)
   python tools/prepare_dialect_data.py \
     --jsonl_file $data_dir/dialect_train.jsonl \
     --output_dir data/Jiang-Huai_train
   
   python tools/prepare_dialect_data.py \
     --jsonl_file $data_dir/dialect_val.jsonl \
     --output_dir data/Jiang-Huai_val
   ```

2. **提取说话人嵌入向量**
   ```bash
   # 生成spk2embedding.pt和utt2embedding.pt
   tools/extract_embedding.py --dir data/Jiang-Huai_train \
     --onnx_path $pretrained_model_dir/campplus.onnx
   ```

3. **提取语音token**
   ```bash
   # 生成utt2speech_token.pt
   tools/extract_speech_token.py --dir data/Jiang-Huai_train \
     --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
   ```

4. **准备parquet格式数据**
   ```bash
   # 创建parquet格式数据，用于高效训练
   mkdir -p data/Jiang-Huai_train/parquet
   tools/make_parquet_list.py --num_utts_per_parquet 100 \
     --num_processes 4 \
     --src_dir data/Jiang-Huai_train \
     --des_dir data/Jiang-Huai_train/parquet
   
   # 创建训练和验证数据列表
   cp data/Jiang-Huai_train/parquet/data.list data/Jiang-Huai_train.list
   cp data/Jiang-Huai_val/parquet/data.list data/Jiang-Huai_val.list
   ```

#### 执行微调

本项目使用`run.sh`脚本执行CosyVoice2模型微调，主要步骤如下：

```bash
# 设置预训练模型和微调后模型路径
pretrained_model_dir=/home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B
finetune_model_dir=/home/chihan/workspace/CosyVoice/pretrained_models/jianghuai

# 设置GPU和分布式训练参数
export CUDA_VISIBLE_DEVICES="2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# 准备方言数据列表
cp /home/chihan/workspace/CosyVoice/data/Jiang-Huai_train/parquet/data.list data/train.data.list
cp /home/chihan/workspace/CosyVoice/data/Jiang-Huai_val/parquet/data.list data/dev.data.list

# 执行分布式训练
# 分别训练文本-语音语言模型(llm)和流匹配合成模块(flow)
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  cosyvoice/bin/train.py \
  --train_engine $train_engine \
  --config /home/chihan/workspace/CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --qwen_pretrain_path /home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model $model \  # llm或flow
  --checkpoint $pretrained_model_dir/$model.pt \
  --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
  --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
  --ddp.dist_backend $dist_backend \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --use_amp \
  --deepspeed_config /home/chihan/workspace/CosyVoice/examples/libritts/cosyvoice2/conf/ds_stage2.json
```

#### 模型平均与导出

微调完成后，执行模型平均以提高稳定性，并导出优化版本用于推理：

```bash
# 模型平均，取验证集上表现最好的5个检查点
average_num=5
python cosyvoice/bin/average_model.py \
  --dst_model $decode_checkpoint \
  --src_path `pwd`/exp/cosyvoice2/$model/$train_engine \
  --num ${average_num} \
  --val_best

# 复制必要文件，构建完整微调模型
cp -r $pretrained_model_dir/* $finetune_model_dir/
cp `pwd`/exp/cosyvoice2/llm/torch_ddp/llm.pt $finetune_model_dir/
cp `pwd`/exp/cosyvoice2/flow/torch_ddp/flow.pt $finetune_model_dir/

# 导出优化版本模型
python cosyvoice/bin/export_jit.py --model_dir $finetune_model_dir
python cosyvoice/bin/export_onnx.py --model_dir $finetune_model_dir
```

#### 低资源适应策略

对于有限的方言语音数据，我们采用以下策略：
- 使用预训练检查点进行初始化，避免从零开始训练
- 应用deepspeed优化和混合精度训练，降低显存需求
- 使用分布式训练加速收敛过程
- 利用模型平均技术提高最终模型稳定性
- 针对方言特性调整配置文件中的模型参数

## 辅助模型组件

系统集成了多个辅助模型，增强整体功能：

1. **语音活动检测模型(VAD)**：基于FSMN架构，精确识别音频中的语音段与非语音段
2. **标点恢复模型**：采用CT-Transformer架构，为识别文本自动添加标点符号
3. **说话人识别模型**：基于CAM++架构，无需预训练即可区分不同说话人
4. **情感识别模块**：利用SenseVoice多任务能力，识别语音中的情感和事件类型

## 支持的文件格式

- **音频**：.mp3, .m4a, .aac, .ogg, .wav, .flac, .wma, .aif
- **视频**：.mp4, .avi, .mov, .mkv

## 技术架构

- **前端**：HTML, CSS, JavaScript
- **后端**：Flask
- **语音识别**：FunASR
- **降噪处理**：UVR5
- **声音克隆**：CosyVoice
- **音频处理**：FFmpeg, PyDub

## 可用模型

- **SenseVoice标准普通话**：通用普通话识别模型
- **江淮方言**：适用于江淮地区方言识别
- **东北方言**：适用于东北地区方言识别
- **西南方言**：适用于西南地区方言识别
- **降噪模型**：HP2, HP5, VR-DeEcho等多种降噪模型

## API接口

系统提供多个API接口用于程序化访问：

- `/api/denoise`：音频降噪处理
- `/upload`：上传和处理音频文件
- `/clone`：声音克隆
- `/models`：获取可用模型列表



## 项目结构
低资源语音包项目框架
```
SenseVoice/
├── app.py          # 主应用入口
├── templates/      # HTML模板
├── static/         # CSS, JavaScript等静态资源
├── uploads/        # 上传文件临时存储
├── outputs/        # 处理结果输出
├── model_cache/    # 模型缓存目录
├── tools/          # 工具模块
│   └── uvr5/       # UVR5降噪工具
├── CosyVoice/      # CosyVoice声音克隆模块
└── finetune.sh     # SenseVoice模型微调脚本
```

```
CosyVoice/
├── scripts/                      # 脚本目录
│   ├── prepare.sh                # 数据准备脚本
│   └── run.sh                    # 训练运行脚本
├── tools/                        # 数据处理工具目录
│   ├── prepare_dialect_data.py   # 方言数据格式转换工具
│   ├── extract_embedding.py      # 提取说话人嵌入向量工具
│   ├── extract_speech_token.py   # 提取语音token工具
│   └── make_parquet_list.py      # 生成parquet格式数据工具
├── data/                         # 数据目录
│   ├── Jiang-Huai_train/         # 江淮方言训练数据
│   ├── Jiang-Huai_val/           # 江淮方言验证数据
│   └── Jiang-Huai_test/          # 江淮方言测试数据
├── pretrained_models/            # 预训练模型目录
│   └── CosyVoice2-0.5B/          # CosyVoice2预训练模型
├── exp/                          # 实验输出目录
│   └── cosyvoice2/               # 模型微调输出
│       ├── llm/                  # 语言模型微调结果
│       └── flow/                 # 合成模型微调结果
├── cosyvoice/                    # CosyVoice核心代码
│   └── bin/                      # 命令行工具
│       ├── train.py              # 训练脚本
│       ├── average_model.py      # 模型平均工具
│       ├── export_jit.py         # JIT模型导出工具
│       └── export_onnx.py        # ONNX模型导出工具
└── examples/                     # 配置示例
    └── libritts/cosyvoice2/conf/ # 配置文件目录
        ├── cosyvoice2.yaml       # 模型配置
        └── ds_stage2.json        # DeepSpeed配置
```

## 故障排除

- **模型加载失败**：确保已下载所有必需的模型文件，并检查模型路径配置
- **处理大文件失败**：尝试增加系统内存或调整batch_size参数
- **降噪效果不佳**：尝试不同的降噪模型和参数组合
- **CUDA错误**：确保已安装兼容的CUDA版本和正确的PyTorch版本

## 未来工作计划

1. **多方言持续扩展与迁移学习策略优化**
   - 引入更多中小方言（如粤语、闽南语、赣语）数据集
   - 引入Prompt-tuning、Adapter等轻量微调机制

2. **语音情感识别与合成的深度融合**
   - 打通情感识别结果与语音合成参数的映射机制
   - 支持情绪驱动的语音回复系统应用场景

3. **系统性能持续优化**
   - 引入TensorRT/ONNX模型加速技术
   - 开展批量音频处理模式与分布式推理测试

4. **前端交互体验升级**
   - 增强前端界面的交互性与结果可视化
   - 支持语音拖拽上传、长音频分段播放与文本编辑功能

5. **部署与应用场景拓展**
   - 尝试将系统部署到边缘设备或小型终端
   - 针对教育、医疗、政务等垂直行业定制语音处理解决方案

## 许可证

本项目遵循MIT许可证。详情请查看LICENSE文件。

## 致谢

本项目使用了以下开源项目：
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui)
- [CosyVoice](https://github.com/thuhcsi/CosyVoice)

## 第三方库说明

本项目部分功能依赖以下第三方库：

- [Matcha-TTS](https://github.com/Matcha-TTS/Matcha-TTS)
  已集成于 `SenseVoice/CosyVoice/third_party/Matcha-TTS` 目录，原项目地址见上方链接。

