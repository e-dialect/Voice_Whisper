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
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice')
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def parse_args():
    parser = argparse.ArgumentParser(description="使用微调的江淮方言模型进行语音合成")
    parser.add_argument('--model_dir', type=str, 
                      default='/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                      help='微调后的江淮方言模型路径')
    parser.add_argument('--text', type=str, default='我想用江淮方言说这句话',
                      help='要合成的文本')
    parser.add_argument('--ref_text', type=str,default= "人人都是张瑞敏一九八四年创业资金",
                      help='参考文本，如果不提供则使用要合成的文本')
    parser.add_argument('--ref_audio', type=str,default='/home/chihan/workspace/SenseVoice/data/KeSpeech_Jiang-Huai/audio/1000057_c027cba7.wav',
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
        
        # 需要清理的文件列表
        model_files = ["llm.pt", "flow.pt"]  # 添加flow.pt
        
        # 清理每个模型文件
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                print(f"检查模型文件: {model_path}")
                # 加载状态字典
                checkpoint = torch.load(model_path, map_location='cpu')
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
                    print(f"清理模型文件中的训练元数据: {model_file}...")
                    torch.save(checkpoint, model_path + ".clean")
                    print(f"已清理模型文件，保存到 {model_path}.clean")
                    # 备份原始文件
                    os.rename(model_path, model_path + ".backup")
                    # 使用清理后的模型文件
                    os.rename(model_path + ".clean", model_path)
                    print(f"模型文件 {model_file} 清理完成")
                
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
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈
        raise RuntimeError(f"无法加载模型: {e}")
        

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
    # 保存音频
    # 添加调试信息
    print(f"音频张量类型: {type(speech)}")
    print(f"音频张量形状: {speech.shape}")
    print(f"采样率: {model.sample_rate}")

    # 确保音频数据格式正确
    try:
        # 方法1: 直接转换并保存
        audio_numpy = speech.detach().cpu().numpy() if torch.is_tensor(speech) else speech.numpy()
        
        # 检查并修复数据形状
        if len(audio_numpy.shape) > 1 and audio_numpy.shape[0] < audio_numpy.shape[1]:
            # 如果是[通道数,长度]格式，转换为[长度,通道数]
            audio_numpy = audio_numpy.T
        
        # 检查和修复无效值
        if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
            print("警告: 检测到无效值，已修复")
            audio_numpy = np.nan_to_num(audio_numpy)
        
        # 确保数据范围正确(-1到1之间)
        max_val = np.max(np.abs(audio_numpy))
        if max_val > 1.0:
            print(f"警告: 音频振幅超出范围，最大值: {max_val}，已归一化")
            audio_numpy = audio_numpy / max_val * 0.9  # 留一点余量
        
        sf.write(args.output, audio_numpy, model.sample_rate)
        print(f"语音合成完成，已保存到: {args.output}")
        
    except Exception as e:
        print(f"使用soundfile保存失败: {e}")
        
        # 备用方法: 使用scipy.io.wavfile
        try:
            import scipy.io.wavfile as wavfile
            # scipy要求音频为int16或float32格式
            audio_numpy = speech.detach().cpu().numpy() if torch.is_tensor(speech) else speech.numpy()
            
            # 确保是float32类型
            audio_numpy = audio_numpy.astype(np.float32)
            
            # 如果数据是二维的，确保形状正确
            if len(audio_numpy.shape) > 1 and audio_numpy.shape[0] < audio_numpy.shape[1]:
                audio_numpy = audio_numpy.T
                
            wavfile.write(args.output, model.sample_rate, audio_numpy)
            print(f"使用scipy.io.wavfile成功保存音频到: {args.output}")
        
        except Exception as e2:
            print(f"两种方法都无法保存音频: {e2}")
            
            # 最后尝试: 使用torchaudio
            try:
                import torchaudio
                if torch.is_tensor(speech):
                    torchaudio.save(args.output, speech.unsqueeze(0) if speech.dim() == 1 else speech, model.sample_rate)
                    print(f"使用torchaudio成功保存音频到: {args.output}")
            except Exception as e3:
                print(f"所有保存方法都失败: {e3}")
                print("请检查音频数据或输出路径")

if __name__ == "__main__":
    main()