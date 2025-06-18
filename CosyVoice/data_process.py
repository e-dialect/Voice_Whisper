import json
import os
import shutil
import torch
import torchaudio
from tqdm import tqdm

# 创建数据目录
os.makedirs("data/northeastern/", exist_ok=True)

# 读取jsonl文件
with open("data/KeSpeech_Northeastern/dialect_train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 创建必要的文件
with open("data/northeastern/wav.scp", "w", encoding="utf-8") as wav_scp, \
     open("data/northeastern/text", "w", encoding="utf-8") as text, \
     open("data/northeastern/utt2spk", "w", encoding="utf-8") as utt2spk:
    
    for line in tqdm(lines):
        item = json.loads(line.strip())
        key = item["key"]
        wav_path = item["source"]
        txt = item["target"]
        
        # 假设所有样本为同一说话人，或者从key中提取说话人ID
        spk = key.split("_")[0]  # 使用key的第一部分作为说话人ID
        
        # 写入wav.scp
        wav_scp.write(f"{key} {wav_path}\n")
        
        # 写入text (添加text_language标记)
        text.write(f"{key} {item['text_language']}{txt}\n")
        
        # 写入utt2spk
        utt2spk.write(f"{key} {spk}\n")

# 生成spk2utt
os.system("tools/spk2utt.py --utt2spk data/northeastern/utt2spk --spk2utt data/northeastern/spk2utt")

print("数据准备完成！")