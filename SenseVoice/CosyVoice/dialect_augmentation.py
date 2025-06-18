import os
import json
import random
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import librosa
import soundfile as sf
from datetime import datetime
import sys
# 添加CosyVoice路径
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice')
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice/third_party/Matcha-TTS')

# CosyVoice导入
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 东北方言特色词汇替换字典
NORTHEASTERN_DICT = {
    "什么": ["啥", "啥玩意儿"],
    "怎么": ["咋", "咋地"],
    "这里": ["这嘎达"],
    "那里": ["那嘎达"],
    "看": ["瞅", "瞧瞧"],
    "很": ["老"],
    "不行": ["不成"],
    "说话": ["唠嗑"],
    "朋友": ["哥们儿", "铁子"],
    "你好": ["嘿 老铁"],
    "工作": ["干活儿"],
    "没有": ["没有啊", "没门儿"],
    "真的": ["真的咧"]
}

# 东北方言语气词
NORTHEASTERN_PARTICLES = ["呗", "咧", "呀", "嘞", "呢", "哎呀"]

class DialectAugmenter:
    def __init__(self, model_path, output_dir, original_jsonl=None):
        """
        初始化方言增强器
        
        Args:
            model_path: CosyVoice模型路径
            output_dir: 输出目录
            original_jsonl: 原始JSONL数据文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载CosyVoice模型
        print(f"加载CosyVoice模型: {model_path}")
        self.model = CosyVoice2(
            model_path, 
            load_jit=False,
            fp16=True if torch.cuda.is_available() else False)
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "augmented_audio"
        self.audio_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载原始数据
        self.original_data = []
        if original_jsonl:
            self.load_jsonl_data(original_jsonl)
        
        # 实验类型配置
        self.experiment_configs = {
            "speed": {
                "values": [0.8, 1.0, 1.2],
                "default": 1.0,
                "name": "语速增强实验"
            },
            "dialect": {
                "values": [False, True],
                "default": False,
                "name": "东北方言文本增强实验"
            },
            "emotion": {
                "values": [False, True],
                "default": False,
                "name": "情感语音增强实验"
            }
        }
            
    def load_jsonl_data(self, jsonl_path):
        """加载JSONL格式的数据"""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith("//"):
                    try:
                        self.original_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析行: {line}")
        
        print(f"加载了 {len(self.original_data)} 条原始数据")
    
    def northeastern_text(self, text):
        """将普通文本转换为东北方言特色文本"""
        result = text
        
        # 替换词汇
        for standard, dialect_options in NORTHEASTERN_DICT.items():
            if standard in result and random.random() < 0.7:  # 70%概率替换
                replacement = random.choice(dialect_options)
                result = result.replace(standard, replacement)
        
        # 添加语气词 (30%概率)
        if random.random() < 0.3 and len(result) > 5:
            if result[-1] in ["。", "!", "?", "！", "？"]:
                result = result[:-1] + random.choice(NORTHEASTERN_PARTICLES) + result[-1]
            else:
                result = result + random.choice(NORTHEASTERN_PARTICLES)
        
        return result
    
    def generate_audio(self, text, output_file, use_emotion=False, speed=1.0):
        """
        生成合成语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            use_emotion: 是否使用情感标签进行合成
            speed: 语速因子 (1.0为正常语速)
        
        Returns:
            输出文件路径和音频时长(毫秒)
        """
        try:
            # 使用CosyVoice2的inference_sft方法生成语音
            # 根据是否使用情感设置instruct_text
            instruct_text = "用开心活泼的语气说" if use_emotion else None
            
            # 获取可用的预训练音色
            available_spks = self.model.list_available_spks()
            speaker = available_spks[0] if available_spks else None
            
            # 设置随机种子
            seed = random.randint(0, 100000)
            torch.manual_seed(seed)
            
            # 生成语音
            for result in self.model.inference_sft(
                text, 
                speaker,
                speed=speed,
                instruct=instruct_text,
                stream=False
            ):
                # 获取生成的音频
                waveform = result['tts_speech']
                
                # 保存音频
                torchaudio.save(
                    output_file,
                    waveform,
                    self.model.sample_rate
                )
                
                # 计算音频时长
                audio_duration_ms = int(len(waveform[0]) / self.model.sample_rate * 1000)
                return output_file, audio_duration_ms
                
        except Exception as e:
            print(f"警告: 生成音频失败: {e}")
            # 创建一个空音频以保持流程完整
            empty_audio = torch.zeros(1, 16000)
            torchaudio.save(output_file, empty_audio, 16000)
            return output_file, 1000
    
    def run_experiments(self, num_samples=100, output_dir=None):
        """
        运行所有三组对照试验
        
        Args:
            num_samples: 要处理的样本数量
            output_dir: 输出目录
        """
        # 确保不超过可用样本数
        num_samples = min(num_samples, len(self.original_data))
        samples_to_process = self.original_data[:num_samples]
        
        if output_dir is None:
            output_dir = self.output_dir
        
        # 为每种实验类型运行实验
        for exp_type in ["speed", "dialect", "emotion"]:
            print(f"\n开始{self.experiment_configs[exp_type]['name']}...")
            self.run_single_experiment(exp_type, samples_to_process, output_dir)
    
    def run_single_experiment(self, experiment_type, samples, output_dir):
        """
        运行单个对照试验
        
        Args:
            experiment_type: 实验类型 ('speed', 'dialect', 'emotion')
            samples: 要处理的样本列表
            output_dir: 输出目录
        """
        # 获取实验配置
        config = self.experiment_configs[experiment_type]
        experiment_name = config["name"]
        variable_values = config["values"]
        
        # 创建输出JSONL文件
        output_jsonl = os.path.join(output_dir, f"augmented_{experiment_type}.jsonl")
        
        # 存储生成的样本
        generated_samples = []
        total_generated = 0
        
        print(f"生成{experiment_name}数据...")
        for idx, sample in enumerate(tqdm(samples)):
            original_text = sample["target"]
            
            # 针对每个变量值生成样本
            for value in variable_values:
                # 对于默认值，仅生成一次样本，避免重复
                if value == config["default"]:
                    # 用原始数据，不需要创建新样本
                    continue
                
                # 根据实验类型设置参数
                use_emotion = value if experiment_type == "emotion" else config["default"]
                dialect_text = value if experiment_type == "dialect" else config["default"]
                speed = value if experiment_type == "speed" else config["default"]
                
                # 处理文本
                text = self.northeastern_text(original_text) if dialect_text else original_text
                
                # 创建唯一ID
                suffix = ""
                if experiment_type == "speed":
                    suffix = f"_speed{int(value*100)}"
                elif experiment_type == "dialect":
                    suffix = "_dialect" if value else ""
                elif experiment_type == "emotion":
                    suffix = "_emotion" if value else ""
                
                unique_id = f"{sample['key']}{suffix}"
                
                # 输出文件路径
                output_file = self.audio_dir / f"{unique_id}.wav"
                
                # 生成语音
                _, duration_ms = self.generate_audio(
                    text, 
                    output_file, 
                    use_emotion=use_emotion, 
                    speed=speed
                )
                
                # 创建新的JSONL条目
                emo_target = "<|HAPPY|>" if use_emotion else "<|NEUTRAL|>"
                new_entry = {
                    "key": unique_id,
                    "text_language": "<|zh|>",
                    "emo_target": emo_target,
                    "event_target": "<|Speech|>",
                    "with_or_wo_itn": "<|woitn|>",
                    "target": text,
                    "source": str(output_file),
                    "target_len": len(text),
                    "source_len": duration_ms,
                    "augmentation_info": {
                        "original_key": sample["key"],
                        "experiment": experiment_type,
                        "dialect_text": dialect_text,
                        "use_emotion": use_emotion,
                        "speed": speed
                    }
                }
                
                # 添加到生成样本列表
                generated_samples.append(new_entry)
                total_generated += 1
                
            # 每10个样本保存一次，避免数据丢失
            if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
                # 保存JSONL文件
                with open(output_jsonl, 'w', encoding='utf-8') as f:
                    # 先写入原始样本
                    for original_sample in samples:
                        f.write(json.dumps(original_sample, ensure_ascii=False) + '\n')
                    
                    # 再写入增强样本
                    for entry in generated_samples:
                        # 移除augmentation_info以保持与原始格式一致
                        output_entry = entry.copy()
                        if "augmentation_info" in output_entry:
                            del output_entry["augmentation_info"]
                        f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
        
        print(f"{experiment_name}完成，原始样本: {len(samples)}，增强样本: {total_generated}")
        print(f"增强数据已保存到: {output_jsonl}")
        
        # 生成实验报告
        self.generate_experiment_report(experiment_type, samples, generated_samples, output_dir)
    
    def generate_experiment_report(self, experiment_type, original_samples, generated_samples, output_dir):
        """生成实验报告"""
        report_file = os.path.join(output_dir, f"{experiment_type}_experiment_report.txt")
        
        config = self.experiment_configs[experiment_type]
        experiment_name = config["name"]
        
        # 写入报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"{experiment_name}报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"原始样本数: {len(original_samples)}\n")
            f.write(f"增强样本数: {len(generated_samples)}\n")
            f.write(f"总样本数: {len(original_samples) + len(generated_samples)}\n\n")
            
            # 根据实验类型添加特定统计
            if experiment_type == "speed":
                speed_stats = {}
                for sample in generated_samples:
                    speed = sample["augmentation_info"]["speed"]
                    speed_stats[speed] = speed_stats.get(speed, 0) + 1
                
                f.write("语速分布:\n")
                for speed, count in sorted(speed_stats.items()):
                    f.write(f"  - {speed}x: {count} 样本\n")
                    
            elif experiment_type == "dialect":
                dialect_count = sum(1 for s in generated_samples if s["augmentation_info"]["dialect_text"])
                f.write(f"东北方言文本样本数: {dialect_count}\n")
                
            elif experiment_type == "emotion":
                emotion_count = sum(1 for s in generated_samples if s["augmentation_info"]["use_emotion"])
                f.write(f"情感增强样本数: {emotion_count}\n")
        
        print(f"实验报告已保存到: {report_file}")


def main():
    """
    python dialect_augmentation.py  --experiment speed --num_samples 50
    """
    parser = argparse.ArgumentParser(description="东北方言语音合成对照试验")
    parser.add_argument("--model", type=str,default="/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice-300M-Instruct", help="CosyVoice模型路径")
    parser.add_argument("--input", type=str, default="/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern/dialect_train.jsonl", help="输入JSONL文件路径")
    parser.add_argument("--output", type=str, default="./augmented_data", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100, help="要处理的样本数量")
    parser.add_argument("--experiment", type=str, choices=["all", "speed", "dialect", "emotion"], 
                        default="speed", help="要运行的实验类型")
    
    args = parser.parse_args()
    
    augmenter = DialectAugmenter(
        model_path=args.model,
        output_dir=args.output,
        original_jsonl=args.input
    )
    
    if args.experiment == "all":
        augmenter.run_experiments(num_samples=args.num_samples, output_dir=args.output)
    else:
        # 运行单个实验
        samples_to_process = augmenter.original_data[:min(args.num_samples, len(augmenter.original_data))]
        augmenter.run_single_experiment(args.experiment, samples_to_process, args.output)


if __name__ == "__main__":
    main()