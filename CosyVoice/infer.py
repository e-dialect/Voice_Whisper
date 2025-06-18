#!/usr/bin/env python
# filepath: /home/chihan/workspace/CosyVoice/synthesize.py
import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
"""
python infer.py \
  --model_dir /home/chihan/workspace/CosyVoice/pretrained_models/jianghuai \
  --text "我想用江淮方言说这句话" \
  --ref_audio /home/chihan/workspace/CosyVoice/asset/zero_shot_prompt.wav \
  --ref_text "希望你以后能够做的比我还好哦" \
  --output output.wav
"""
# 添加CosyVoice路径

# 添加CosyVoice路径
sys.path.append('/home/chihan/workspace/CosyVoice')
sys.path.append('/home/chihan/workspace/CosyVoice/third_party/Matcha-TTS')


from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def parse_args():
    parser = argparse.ArgumentParser(description="使用微调的江淮方言模型进行语音合成")
    parser.add_argument('--model_dir', type=str, 
                      default='/home/chihan/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                      help='微调后的江淮方言模型路径')
    parser.add_argument('--text', type=str, default='我想用江淮方言说这句话',
                      help='要合成的文本')
    parser.add_argument('--ref_text', type=str,default= "希望你以后能够做的比我还好哦",
                      help='参考文本，如果不提供则使用要合成的文本')
    parser.add_argument('--ref_audio', type=str,default='/home/chihan/workspace/CosyVoice/asset/zero_shot_prompt.wav',
                      help='参考音频文件路径')
    parser.add_argument('--output', type=str, default='output.wav',
                      help='输出音频文件路径')
    parser.add_argument('--speed', type=float, default=1.0,
                      help='语速因子，大于1加快，小于1减慢')
    return parser.parse_args()

def load_model(model_dir):
    """加载微调的CosyVoice模型"""
    print(f"加载模型: {model_dir}")
    try:
        # 检查配置文件是否存在
        config_path = os.path.join(model_dir, "cosyvoice2.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, "cosyvoice.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: 既没有cosyvoice2.yaml也没有cosyvoice.yaml")
            
        # 尝试预处理模型文件（移除训练元数据）
        llm_path = os.path.join(model_dir, "llm.pt")
        if os.path.exists(llm_path):
            # 加载状态字典
            checkpoint = torch.load(llm_path, map_location='cpu')
            need_save = False
            
            # 检查是否包含训练元数据
            if isinstance(checkpoint, dict):
                # 直接移除顶层的训练元数据键
                if "epoch" in checkpoint:
                    del checkpoint["epoch"]
                    need_save = True
                if "step" in checkpoint:
                    del checkpoint["step"]
                    need_save = True
                
                # 如果存在model子键，提取它
                if "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                    need_save = True
                    
            # 如果进行了任何修改，保存清理后的模型
            if need_save:
                print("清理模型文件中的训练元数据...")
                torch.save(checkpoint, llm_path + ".clean")
                print(f"已清理模型文件，保存到 {llm_path}.clean")
                # 备份原始文件
                os.rename(llm_path, llm_path + ".backup")
                # 使用清理后的模型文件
                os.rename(llm_path + ".clean", llm_path)
                print("模型文件清理完成")
                
        # 加载模型
        model = CosyVoice2(
            model_dir,
            load_jit=False,
            load_trt=False,
            fp16=False
        )
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise RuntimeError(f"无法加载模型: {e}")  # 确保错误被正确传播
        

def main():
    args = parse_args()
    
    # 加载模型
    model = load_model(args.model_dir)
    print("模型加载成功")
    
    # 准备参考文本
    ref_text = args.ref_text if args.ref_text else args.text
    
    # 加载参考音频
    ref_speech = load_wav(args.ref_audio, 16000)
    
    # 生成语音
    print(f"正在合成文本: {args.text}")
    print(f"参考文本: {ref_text}")
    print(f"参考音频: {args.ref_audio}")
    print(f"语速因子: {args.speed}")
    
    generator = model.inference_zero_shot(
        args.text,      # 要合成的文本
        ref_text,       # 参考文本
        ref_speech,     # 参考音频
        stream=False,
        speed=args.speed
    )
    
    result = next(generator)
    speech = result['tts_speech']
    
    # 保存音频
    sf.write(args.output, speech.numpy(), model.sample_rate)
    print(f"语音合成完成，已保存到: {args.output}")

if __name__ == "__main__":
    main()