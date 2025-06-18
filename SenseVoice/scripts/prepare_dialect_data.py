import json
import os
import argparse
import random
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

def get_audio_duration_ms(audio_path):
    """
    获取音频文件的持续时间（毫秒）
    """
    try:
        if os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=None)
            duration_ms = int(len(y) / sr * 1000)
            return duration_ms
        else:
            return None
    except Exception as e:
        print(f"警告: 处理音频文件 {audio_path} 时出错: {e}")
        return None

def prepare_dialect_data(input_file, output_dir, audio_dir, language="zh", 
                         val_size=0.1, test_size=0.1, emo_label="<|NEUTRAL|>"):
    """
    将方言数据转换为SenseVoice训练格式，并划分训练集、验证集和测试集
    
    Args:
        input_file: 输入文件，每行格式为"音频文件,文本"
        output_dir: 输出目录，保存训练、验证和测试数据
        audio_dir: 音频文件所在目录
        language: 语言代码，默认为zh（中文）
        val_size: 验证集比例，默认为0.1（10%）
        test_size: 测试集比例，默认为0.1（10%）
        emo_label: 情感标签，默认为<|NEUTRAL|>
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "dialect_train.jsonl")
    val_file = os.path.join(output_dir, "dialect_val.jsonl")
    test_file = os.path.join(output_dir, "dialect_test.jsonl")  # 新增测试集文件
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 去重处理
    unique_audio_paths = set()
    unique_lines = []
    duplicates = 0
    
    for line in lines:
        if not line.strip() or ',' not in line:
            continue
        
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
            
        audio_file, text = parts
        audio_file = audio_file.strip()
        
        # 构建绝对路径
        audio_basename = os.path.basename(audio_file)
        if not audio_basename.lower().endswith('.wav'):
            audio_basename += '.wav'
        
        abs_path = os.path.abspath(os.path.join(audio_dir, audio_basename))
        
        # 检查是否重复
        if abs_path in unique_audio_paths:
            duplicates += 1
            continue
        
        unique_audio_paths.add(abs_path)
        unique_lines.append(line)
    
    print(f"原始数据行数: {len(lines)}")
    print(f"去除重复后行数: {len(unique_lines)}")
    print(f"去除的重复项: {duplicates}")
    
    # 使用去重后的数据继续处理
    lines = unique_lines
    
    # 随机打乱数据
    random.seed(42)  # 设置随机种子确保可复现性
    random.shuffle(lines)
    
    # 划分训练集、验证集和测试集
    total = len(lines)
    test_idx = int(total * (1 - test_size))
    val_idx = int(total * (1 - test_size - val_size))
    
    train_lines = lines[:val_idx]
    val_lines = lines[val_idx:test_idx]
    test_lines = lines[test_idx:]
    
    print(f"总数据条数: {total}")
    print(f"训练集条数: {len(train_lines)} ({len(train_lines)/total:.1%})")
    print(f"验证集条数: {len(val_lines)} ({len(val_lines)/total:.1%})")
    print(f"测试集条数: {len(test_lines)} ({len(test_lines)/total:.1%})")
    
    # 处理训练集
    train_data = process_data_lines(train_lines, audio_dir, language, emo_label)
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 处理验证集
    val_data = process_data_lines(val_lines, audio_dir, language, emo_label)
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 处理测试集
    test_data = process_data_lines(test_lines, audio_dir, language, emo_label)
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"训练集已保存至: {train_file}")
    print(f"验证集已保存至: {val_file}")
    print(f"测试集已保存至: {test_file}")
    
    # 生成数据统计报告
    generate_statistics(train_data, val_data, test_data)

def process_data_lines(lines, audio_dir, language, emo_label):
    """处理数据行并生成SenseVoice格式的条目"""
    data = []
    skipped = 0  # 跳过的记录数
    
    for line in tqdm(lines, desc="处理数据"):
        if not line.strip() or ',' not in line:
            skipped += 1
            continue
            
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            print(f"警告: 无效的行格式: '{line.strip()}'")
            skipped += 1
            continue
            
        audio_file, text = parts
        audio_file = audio_file.strip()  # 去除前后空白
        
        # 跳过明显是占位符的记录
        if audio_file == "音频文件" or "音频文件" in audio_file:
            print(f"跳过占位符记录: {line.strip()}")
            skipped += 1
            continue
        
        # 移除任何路径部分，并提取正确的文件名
        audio_basename = os.path.basename(audio_file)
        # 移除任何扩展名作为key
        key = Path(audio_basename).stem
        
        # 确保音频文件名包含.wav扩展名
        if not audio_basename.lower().endswith('.wav'):
            audio_basename += '.wav'
        
        # 构建完整的音频路径
        source_path = os.path.abspath(os.path.join(audio_dir, audio_basename))
        
        # 检查音频文件是否存在
        if not os.path.isfile(source_path):
            print(f"警告: 音频文件不存在 - {source_path}")
            skipped += 1
            continue
        
        # 获取音频实际长度（毫秒）
        source_len = get_audio_duration_ms(source_path)
        if source_len is None or source_len <= 0:
            skipped += 1
            continue
        
        entry = {
            "key": key,
            "text_language": f"<|{language}|>",
            "emo_target": emo_label,
            "event_target": "<|Speech|>",
            "with_or_wo_itn": "<|woitn|>",
            "target": text,
            "source": source_path,
            "target_len": len(text),
            "source_len": source_len
        }
        data.append(entry)
    
    print(f"成功处理 {len(data)} 条记录，跳过 {skipped} 条无效记录")
    return data

def generate_statistics(train_data, val_data, test_data):
    """生成数据统计报告，包括测试集"""
    all_data = train_data + val_data + test_data
    
    # 计算音频总时长
    total_duration_ms = sum(item["source_len"] for item in all_data)
    train_duration_ms = sum(item["source_len"] for item in train_data)
    val_duration_ms = sum(item["source_len"] for item in val_data)
    test_duration_ms = sum(item["source_len"] for item in test_data)
    
    # 计算文本总长度
    total_text_len = sum(item["target_len"] for item in all_data)
    avg_text_len = total_text_len / len(all_data) if all_data else 0
    
    # 统计报告
    print("\n========== 数据统计 ==========")
    print(f"总样本数: {len(all_data)}")
    print(f"总音频时长: {total_duration_ms/60000:.2f}分钟")
    print(f"  训练集: {train_duration_ms/60000:.2f}分钟")
    print(f"  验证集: {val_duration_ms/60000:.2f}分钟")
    print(f"  测试集: {test_duration_ms/60000:.2f}分钟")
    print(f"平均每条文本长度: {avg_text_len:.2f}字符")
    print("==============================\n")

def verify_data(jsonl_file, num_samples=5):
    """验证JSONL数据文件的正确性，并展示样例"""
    print(f"\n验证文件: {jsonl_file}")
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("警告: 文件为空!")
            return
            
        print(f"文件包含 {len(lines)} 条记录")
        
        # 随机抽取样本进行展示
        samples = random.sample(lines, min(num_samples, len(lines)))
        print(f"\n随机抽取 {len(samples)} 条记录:")
        
        for i, sample in enumerate(samples):
            try:
                data = json.loads(sample)
                print(f"\n样本 {i+1}:")
                print(f"  Key: {data['key']}")
                print(f"  文本: {data['target']}")
                print(f"  语言: {data['text_language']}")
                print(f"  音频路径: {data['source']}")
                print(f"  音频长度: {data['source_len']/1000:.2f}秒")
                
                # 检查音频文件是否存在
                if not os.path.exists(data['source']):
                    print(f"  警告: 音频文件不存在!")
            except Exception as e:
                print(f"  解析错误: {e}")
    
    except Exception as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备方言数据用于SenseVoice模型微调')
    parser.add_argument('--input', required=True, help='输入文本文件，每行格式为"音频文件,文本"')
    parser.add_argument('--output_dir', required=True, help='输出目录，用于保存训练和验证数据')
    parser.add_argument('--audio_dir', required=True, help='音频文件所在目录的路径')
    parser.add_argument('--language', default='zh', help='语言代码，默认为zh（中文）')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例，默认为0.1（10%）')
    parser.add_argument('--test_size', type=float, default=0.1, help='测试集比例，默认为0.1（10%）')
    parser.add_argument('--emo_label', default='<|NEUTRAL|>', 
                      choices=['<|NEUTRAL|>', '<|EMO_UNKNOWN|>'], 
                      help='情感标签，默认为<|NEUTRAL|>')
    parser.add_argument('--verify', action='store_true', help='生成数据后进行验证')
    
    args = parser.parse_args()
    
    # 准备数据
    prepare_dialect_data(
        args.input, 
        args.output_dir, 
        args.audio_dir, 
        args.language, 
        args.val_size,
        args.test_size, 
        args.emo_label
    )
    
    # 验证生成的数据
    if args.verify:
        train_file = os.path.join(args.output_dir, "dialect_train.jsonl")
        val_file = os.path.join(args.output_dir, "dialect_val.jsonl")
        test_file = os.path.join(args.output_dir, "dialect_test.jsonl")
        verify_data(train_file)
        verify_data(val_file)
        verify_data(test_file)