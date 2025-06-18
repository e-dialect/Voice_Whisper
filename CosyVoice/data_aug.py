#!/usr/bin/env python
# filepath: /home/chihan/workspace/CosyVoice/jsonl_data_augment.py
import os
import sys
import argparse
import torch
import torchaudio
import random
import json
import uuid
from tqdm import tqdm
# 指定使用的GPU设备号，例如使用第0个GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
"""
python data_aug.py \
  --input_jsonl /home/chihan/workspace/CosyVoice/data/KeSpeech_Jiang-Huai/dialect_train.jsonl \
  --output_dir /home/chihan/workspace/CosyVoice/data/KeSpeech_Jiang-Huai_augmented \
  --audio_dir /home/chihan/workspace/CosyVoice/data/KeSpeech_Jiang-Huai_augmented/audio \
  --speed_range 0.9,1.1
"""
# 添加CosyVoice路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description="使用CosyVoice进行江淮方言JSONL格式数据增强")
    parser.add_argument('--model_dir', type=str, default='/home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                      help='微调后的江淮方言模型路径')
    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='输入的JSONL数据文件')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出数据目录')
    parser.add_argument('--audio_dir', type=str, required=True,
                      help='增强音频保存目录')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--speed_range', type=str, default='0.9,1.1',
                      help='语速调整范围，格式为min,max')
    return parser.parse_args()

def read_jsonl_data(jsonl_file):
    """读取JSONL格式的数据"""
    data_items = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data_items.append(item)
    return data_items

def get_audio_length(audio_tensor, sample_rate):
    """获取音频长度（帧数）"""
    return audio_tensor.shape[1]

def load_cosyvoice_model(model_dir):
    """加载CosyVoice模型，使用更稳定的方法"""
    print(f"尝试加载CosyVoice2模型: {model_dir}")
    try:
        # 使用与Flask应用相同的参数进行模型初始化
        model = CosyVoice2(
            model_dir,
            load_jit=False,  # 不使用JIT版本
            load_trt=False,  # 不使用TensorRT
            fp16=False       # 不使用半精度
        )
        print(f"成功使用CosyVoice2加载模型: {model_dir}")
        return model
    except Exception as e1:
        print(f"CosyVoice2加载失败: {e1}")
        try:
            # 尝试使用基础CosyVoice类加载
            print("尝试使用CosyVoice加载模型...")
            from cosyvoice.cli.cosyvoice import CosyVoice
            model = CosyVoice(model_dir)
            print(f"成功使用CosyVoice加载模型: {model_dir}")
            return model
        except Exception as e2:
            raise RuntimeError(f"模型加载失败:\nCosyVoice2错误: {e1}\nCosyVoice错误: {e2}")

def generate_speech(cosyvoice_model, text, ref_text, ref_audio_path, speed=1.0):
    """使用原始数据作为参考进行zero-shot语音合成
    
    参数:
        cosyvoice_model: CosyVoice模型
        text: 要合成的文本
        ref_text: 参考文本(原始音频的转录文本)
        ref_audio_path: 参考音频路径
        speed: 语速因子
    """
    try:
        print(f"使用zero-shot方式生成语音，文本: {text[:30]}...")
        print(f"参考文本: {ref_text[:30]}...")
        print(f"参考音频: {ref_audio_path}")
        
        # 加载参考音频
        ref_speech = load_wav(ref_audio_path, 16000)
        
        # 使用原始数据作为参考进行zero-shot语音合成
        generator = cosyvoice_model.inference_zero_shot(
            text,          # 要合成的文本
            ref_text,      # 使用原始数据的转录文本作为参考
            ref_speech,    # 使用原始数据的音频作为参考
            stream=False,
            speed=speed    # 传入速度参数
        )
        result = next(generator)
        return result['tts_speech']
        
    except Exception as e:
        raise RuntimeError(f"零样本语音生成失败: {e}")

def augment_jsonl_data(cosyvoice_model, data_items, audio_dir, seed=42, speed_range=(0.9, 1.1)):
    """使用CosyVoice模型对JSONL格式数据进行增强"""
    os.makedirs(audio_dir, exist_ok=True)
    augmented_items = []
    
    for i, item in enumerate(tqdm(data_items, desc="正在进行数据增强")):
        # 设置随机种子
        current_seed = seed + i
        set_all_random_seed(current_seed)
        
        # 提取文本和原始音频路径
        text = item["target"]
        ref_text = item["target"]  # 使用相同文本作为参考
        ref_audio_path = item["source"]  # 原始音频路径
        
        # 检查参考音频是否存在
        if not os.path.exists(ref_audio_path):
            print(f"参考音频不存在，跳过: {ref_audio_path}")
            continue
        
        # 随机选择语速
        speed = random.uniform(speed_range[0], speed_range[1])
        
        # 生成唯一ID作为文件名
        unique_id = f"{uuid.uuid4().hex[:8]}_{item['key'].split('_')[1]}"
        new_audio_path = os.path.join(audio_dir, f"{unique_id}.wav")
        
        try:
            # 使用原始数据作为参考生成语音
            speech = generate_speech(cosyvoice_model, text, ref_text, ref_audio_path, speed)
            
            # 保存生成的音频
            speech_tensor = torch.tensor(speech.numpy().flatten()).unsqueeze(0)
            torchaudio.save(new_audio_path, speech_tensor, cosyvoice_model.sample_rate)
            
            # 获取音频长度
            audio_length = get_audio_length(speech_tensor, cosyvoice_model.sample_rate)
            
            # 创建新的JSONL条目
            new_item = item.copy()
            new_item["key"] = unique_id
            new_item["source"] = new_audio_path
            new_item["source_len"] = audio_length
            
            # 添加到增强数据列表
            augmented_items.append(new_item)
        except Exception as e:
            print(f"处理 {item['key']} 时出错: {str(e)}")
    
    return augmented_items

def save_jsonl_data(data_list, output_path):
    """保存数据为JSONL格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    args = parse_args()
    
    # 解析语速范围
    min_speed, max_speed = map(float, args.speed_range.split(','))
    speed_range = (min_speed, max_speed)
    
    # 使用改进的模型加载函数
    try:
        cosyvoice_model = load_cosyvoice_model(args.model_dir)
        print(f"成功加载模型: {args.model_dir}")
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取原始JSONL数据
    original_data = read_jsonl_data(args.input_jsonl)
    print(f"从 {args.input_jsonl} 加载了 {len(original_data)} 条原始数据")
    
    # 进行数据增强
    augmented_data = augment_jsonl_data(
        cosyvoice_model, 
        original_data, 
        args.audio_dir, 
        args.seed,
        speed_range
    )
    
    print(f"成功生成 {len(augmented_data)} 条增强数据")
    
    # 保存增强数据
    augmented_jsonl_path = os.path.join(args.output_dir, "augmented_data.jsonl")
    save_jsonl_data(augmented_data, augmented_jsonl_path)
    
    # 保存合并后的数据
    combined_data = original_data + augmented_data
    combined_jsonl_path = os.path.join(args.output_dir, "combined_data.jsonl")
    save_jsonl_data(combined_data, combined_jsonl_path)
    
    print(f"数据增强完成！")
    print(f"增强数据已保存至: {augmented_jsonl_path}")
    print(f"合并数据已保存至: {combined_jsonl_path}")
    print(f"数据总量从 {len(original_data)} 增加到 {len(combined_data)} 条")

if __name__ == "__main__":
    main()