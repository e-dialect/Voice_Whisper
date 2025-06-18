import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa
import argparse
from collections import Counter
import pandas as pd


# 创建图表保存目录
PLOT_DIR = '/home/chihan/workspace/SenseVoice/data/KeSpeech_Statistics'
os.makedirs(PLOT_DIR, exist_ok=True)

def analyze_dataset(dataset_path, audio_base_path=None, check_audio=True, dataset_type='unknown'):
    """
    对JSONL格式的语音数据集进行详细统计分析
    
    参数:
    dataset_path: JSONL文件路径
    audio_base_path: 音频文件的基础路径（在本例中不需要，因为使用绝对路径）
    check_audio: 是否检查音频文件实际时长
    dataset_type: 数据集类型（train, val, test）用于图表命名
    """
    # 统计变量初始化
    total_samples = 0
    total_duration = 0
    text_lengths = []
    audio_durations = []
    missing_audios = 0
    dialects = Counter()  # 如果数据集中包含方言标签
    speakers = Counter()  # 如果数据集中包含说话人标签
    emotions = Counter()  # 情感分布统计
    
    # 读取JSONL文件
    print(f"正在分析数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        try:
            # 解析JSONL
            data = json.loads(line.strip())
            total_samples += 1
            
            # 提取文本长度 (使用target字段)
            if 'target' in data:
                text_lengths.append(len(data['target']))
            
            # 提取音频信息 (使用source字段)
            if 'source' in data:
                audio_path = data['source']  # 已经是绝对路径
                
                # 检查音频文件是否存在
                if os.path.exists(audio_path):
                    # 获取音频时长
                    try:
                        # 直接加载音频获取实际时长
                        duration = librosa.get_duration(filename=audio_path)
                        audio_durations.append(duration)
                        total_duration += duration
                    except Exception as e:
                        print(f"无法读取音频 {audio_path}: {e}")
                else:
                    missing_audios += 1
                    print(f"找不到音频文件: {audio_path}")
            
            # 统计情感信息
            if 'emo_target' in data:
                emotions[data['emo_target']] += 1
            
            # 统计方言信息（如果有）
            if 'dialect' in data:
                dialects[data['dialect']] += 1
            
            # 统计说话人信息（如果有）
            if 'speaker' in data:
                speakers[data['speaker']] += 1
                
        except json.JSONDecodeError:
            print(f"警告: 无法解析行: {line[:50]}...")
        except Exception as e:
            print(f"处理样本时出错: {e}")
    
    # 计算统计结果
    results = {
        "总样本数": total_samples,
        "总音频时长(分钟)": total_duration / 60,
        "平均音频时长(秒)": total_duration / total_samples if total_samples > 0 else 0,
        "平均文本长度(字符)": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "最短文本长度": min(text_lengths) if text_lengths else 0,
        "最长文本长度": max(text_lengths) if text_lengths else 0,
        "文本长度中位数": np.median(text_lengths) if text_lengths else 0,
        "缺失音频文件数": missing_audios,
        # 添加原始数据，用于后续整合分析
        "text_lengths": text_lengths,
        "audio_durations": audio_durations,
        "emotions": emotions,
        "dialects": dialects,
        "speakers": speakers
    }
    
    # 获取数据集名称用于图表标题
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    
    # 绘制文本长度分布直方图 - 使用英文
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50, alpha=0.7, color='blue')
    plt.title(f'{dataset_type.capitalize()} Set - Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # 保存图表到统一目录
    plt.savefig(os.path.join(PLOT_DIR, f"{dataset_type}_text_length_dist.png"))
    plt.close()
    
    # 如果有音频时长信息，绘制音频时长分布 - 使用英文
    if audio_durations:
        plt.figure(figsize=(10, 6))
        plt.hist(audio_durations, bins=50, alpha=0.7, color='green')
        plt.title(f'{dataset_type.capitalize()} Set - Audio Duration Distribution')
        plt.xlabel('Audio Duration (seconds)')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, f"{dataset_type}_audio_duration_dist.png"))
        plt.close()
    
    # 如果有情感信息，绘制情感分布饼图
    if emotions:
        plt.figure(figsize=(12, 8))
        emotion_labels = [e.replace('<|', '').replace('|>', '') for e in emotions.keys()]
        plt.pie(emotions.values(), labels=emotion_labels, autopct='%1.1f%%')
        plt.title(f'{dataset_type.capitalize()} Set - Emotion Distribution')
        plt.savefig(os.path.join(PLOT_DIR, f"{dataset_type}_emotion_dist.png"))
        plt.close()
    
    # 如果有方言信息，绘制方言分布饼图
    if dialects:
        plt.figure(figsize=(12, 8))
        plt.pie(dialects.values(), labels=dialects.keys(), autopct='%1.1f%%')
        plt.title(f'{dataset_type.capitalize()} Set - Dialect Distribution')
        plt.savefig(os.path.join(PLOT_DIR, f"{dataset_type}_dialect_dist.png"))
        plt.close()
    
    # 打印结果
    print(f"\n========== {dataset_type.upper()} 数据集统计 ==========")
    for key, value in results.items():
        if key not in ["text_lengths", "audio_durations", "emotions", "dialects", "speakers"]:
            print(f"{key}: {value}")
    
    # 如果有情感信息，打印情感分布
    if emotions:
        print("\n情感分布:")
        for emotion, count in emotions.most_common():
            print(f"  {emotion}: {count} 样本 ({count/total_samples*100:.2f}%)")
    
    # 如果有方言信息，打印方言分布
    if dialects:
        print("\n方言分布:")
        for dialect, count in dialects.most_common():
            print(f"  {dialect}: {count} 样本 ({count/total_samples*100:.2f}%)")
    
    # 如果有说话人信息，打印说话人统计
    if speakers:
        print(f"\n说话人总数: {len(speakers)}")
        print(f"每个说话人平均样本数: {total_samples/len(speakers):.2f}")
        top_speakers = speakers.most_common(10)
        print("Top 10 说话人:")
        for speaker, count in top_speakers:
            print(f"  {speaker}: {count} 样本")
    
    print("==============================")
    
    return results

def create_combined_visualizations(train_stats, val_stats, test_stats):
    """创建整合的数据可视化"""
    print("生成整合数据可视化...")
    
    # 数据集划分图
    train_samples = train_stats['总样本数']
    val_samples = val_stats['总样本数']
    test_samples = test_stats['总样本数']
    
    plt.figure(figsize=(10, 6))
    plt.pie([train_samples, val_samples, test_samples], 
            labels=['Training', 'Validation', 'Test'], 
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Dataset Split Distribution')
    plt.savefig(os.path.join(PLOT_DIR, "dataset_split_dist.png"))
    plt.close()
    
    # 合并文本长度数据
    all_text_lengths = []
    all_text_lengths.extend(train_stats['text_lengths'])
    all_text_lengths.extend(val_stats['text_lengths'])
    all_text_lengths.extend(test_stats['text_lengths'])
    
    # 整体文本长度分布
    plt.figure(figsize=(12, 6))
    plt.hist(all_text_lengths, bins=50, alpha=0.7, color='purple')
    plt.title('Overall Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "overall_text_length_dist.png"))
    plt.close()
    
    # 合并音频时长数据
    all_audio_durations = []
    all_audio_durations.extend(train_stats['audio_durations'])
    all_audio_durations.extend(val_stats['audio_durations'])
    all_audio_durations.extend(test_stats['audio_durations'])
    
    # 整体音频时长分布
    plt.figure(figsize=(12, 6))
    plt.hist(all_audio_durations, bins=50, alpha=0.7, color='teal')
    plt.title('Jiang-Huai Overall Audio Duration Distribution')
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "overall_audio_duration_dist.png"))
    plt.close()
    
    # 合并情感数据
    all_emotions = Counter()
    all_emotions.update(train_stats['emotions'])
    all_emotions.update(val_stats['emotions'])
    all_emotions.update(test_stats['emotions'])
    
    # 整体情感分布
    if all_emotions:
        plt.figure(figsize=(12, 8))
        emotion_labels = [e.replace('<|', '').replace('|>', '') for e in all_emotions.keys()]
        plt.pie([count for _, count in all_emotions.most_common()], 
                labels=emotion_labels, 
                autopct='%1.1f%%')
        plt.title('Overall Emotion Distribution')
        plt.savefig(os.path.join(PLOT_DIR, "overall_emotion_dist.png"))
        plt.close()
    
    # 创建综合数据表格
    summary_data = {
        "Metric": ["Total Samples", "Total Duration (min)", "Avg Duration (sec)", "Avg Text Length"],
        "Training": [
            train_stats['总样本数'],
            train_stats['总音频时长(分钟)'],
            train_stats['平均音频时长(秒)'],
            train_stats['平均文本长度(字符)']
        ],
        "Validation": [
            val_stats['总样本数'],
            val_stats['总音频时长(分钟)'],
            val_stats['平均音频时长(秒)'],
            val_stats['平均文本长度(字符)']
        ],
        "Test": [
            test_stats['总样本数'],
            test_stats['总音频时长(分钟)'],
            test_stats['平均音频时长(秒)'],
            test_stats['平均文本长度(字符)']
        ],
        "Overall": [
            train_stats['总样本数'] + val_stats['总样本数'] + test_stats['总样本数'],
            train_stats['总音频时长(分钟)'] + val_stats['总音频时长(分钟)'] + test_stats['总音频时长(分钟)'],
            sum(all_audio_durations) / len(all_audio_durations) if all_audio_durations else 0,
            sum(all_text_lengths) / len(all_text_lengths) if all_text_lengths else 0
        ]
    }
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(PLOT_DIR, "dataset_summary.csv"), index=False)
    
    # 绘制汇总表格作为图像
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Dataset Summary Statistics')
    plt.savefig(os.path.join(PLOT_DIR, "dataset_summary_table.png"), bbox_inches='tight', dpi=200)
    plt.close()
    
    # 创建综合展示图 - 四合一面板
    fig = plt.figure(figsize=(20, 16))
    
    # 数据集划分
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.pie([train_samples, val_samples, test_samples], 
            labels=['Training', 'Validation', 'Test'], 
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff','#99ff99'])
    ax1.set_title('Dataset Split Distribution')
    
    # 文本长度分布
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(all_text_lengths, bins=50, alpha=0.7, color='purple')
    ax2.set_title('Text Length Distribution')
    ax2.set_xlabel('Characters')
    ax2.set_ylabel('Samples')
    ax2.grid(True, alpha=0.3)
    
    # 音频时长分布
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(all_audio_durations, bins=50, alpha=0.7, color='teal')
    ax3.set_title('Audio Duration Distribution')
    ax3.set_xlabel('Seconds')
    ax3.set_ylabel('Samples')
    ax3.grid(True, alpha=0.3)
    
    # 情感分布
    if all_emotions:
        ax4 = fig.add_subplot(2, 2, 4)
        emotion_labels = [e.replace('<|', '').replace('|>', '') for e in all_emotions.keys()]
        ax4.pie([count for _, count in all_emotions.most_common()], 
                labels=emotion_labels, 
                autopct='%1.1f%%')
        ax4.set_title('Emotion Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "dataset_overview.png"), dpi=200)
    plt.close()
    
    print(f"所有图表已保存到: {PLOT_DIR}")

def main():
    global PLOT_DIR  # 在函数开头声明
    parser = argparse.ArgumentParser(description='对语音数据集进行统计分析')
    parser.add_argument('--train', type=str, help='训练集JSONL文件路径', 
                        default='/home/chihan/workspace/SenseVoice/data/KeSpeech_Jiang-Huai/dialect_train.jsonl')
    parser.add_argument('--val', type=str, help='验证集JSONL文件路径',
                        default='/home/chihan/workspace/SenseVoice/data/KeSpeech_Jiang-Huai/dialect_val.jsonl')
    parser.add_argument('--test', type=str, help='测试集JSONL文件路径',
                        default='/home/chihan/workspace/SenseVoice/data/KeSpeech_Jiang-Huai/dialect_test.jsonl')
    parser.add_argument('--audio_path', type=str, help='音频文件基础路径 (对于绝对路径可忽略)')
    parser.add_argument('--no_check_audio', action='store_true', help='不检查实际音频时长(更快但不准确)')
    parser.add_argument('--output_dir', type=str, help='图表输出目录', 
                        default=PLOT_DIR)
    
    args = parser.parse_args()
    
    # 更新全局输出目录
    PLOT_DIR = args.output_dir
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 反转check_audio的逻辑，默认为True
    check_audio = not args.no_check_audio
    
    # 分析训练集
    if os.path.exists(args.train):
        train_stats = analyze_dataset(args.train, args.audio_path, check_audio, 'train')
    else:
        print(f"警告: 训练集文件不存在 - {args.train}")
        return
    
    # 分析验证集
    if os.path.exists(args.val):
        val_stats = analyze_dataset(args.val, args.audio_path, check_audio, 'val')
    else:
        print(f"警告: 验证集文件不存在 - {args.val}")
        return
    
    # 分析测试集
    if os.path.exists(args.test):
        test_stats = analyze_dataset(args.test, args.audio_path, check_audio, 'test')
    else:
        print(f"警告: 测试集文件不存在 - {args.test}")
        return
    
    # 汇总统计
    if 'train_stats' in locals() and 'val_stats' in locals() and 'test_stats' in locals():
        total_samples = train_stats['总样本数'] + val_stats['总样本数'] + test_stats['总样本数']
        total_duration = train_stats['总音频时长(分钟)'] + val_stats['总音频时长(分钟)'] + test_stats['总音频时长(分钟)']
        
        print("\n========== 数据集总体统计 ==========")
        print(f"总样本数: {total_samples}")
        print(f"总音频时长: {total_duration:.2f}分钟")
        print(f"  训练集: {train_stats['总音频时长(分钟)']:.2f}分钟 ({train_stats['总样本数']} 样本)")
        print(f"  验证集: {val_stats['总音频时长(分钟)']:.2f}分钟 ({val_stats['总样本数']} 样本)")
        print(f"  测试集: {test_stats['总音频时长(分钟)']:.2f}分钟 ({test_stats['总样本数']} 样本)")
        print("==================================")
        
        # 创建整合的可视化
        create_combined_visualizations(train_stats, val_stats, test_stats)

if __name__ == "__main__":
    main()