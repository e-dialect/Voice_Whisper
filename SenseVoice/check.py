import os
import subprocess
import json
from tqdm import tqdm
import re

# 数据集路径
dataset_path = "/home/chihan/workspace/SenseVoice/data/KeSpeech_Jiang-Huai/audio"

# 获取音频时长的函数
def get_audio_duration(file_path):
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = json.loads(result.stdout)
    return float(output['format']['duration'])

# 初始化变量
total_duration = 0
speakers = set()

# 查找所有音频文件
audio_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".wav", ".flac", ".mp3")):  # 常见音频格式
            audio_files.append(os.path.join(root, file))

print(f"找到 {len(audio_files)} 个音频文件")

# 处理每个音频文件
for file_path in tqdm(audio_files, desc="处理音频文件"):
    # 获取时长
    try:
        duration = get_audio_duration(file_path)
        total_duration += duration
        
        # 提取说话人ID (根据实际数据集结构调整)
        # 通常说话人ID会在文件名或目录结构中
        filename = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        
        # 假设说话人ID在父目录或文件名前缀中
        # 这部分需要根据实际数据集结构调整
        potential_speaker_id = parent_dir
        if re.match(r'[a-zA-Z\d]+', filename.split('_')[0]):
            potential_speaker_id = filename.split('_')[0]
            
        speakers.add(potential_speaker_id)
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 打印结果
print(f"总音频时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")
print(f"说话人数量: {len(speakers)}")