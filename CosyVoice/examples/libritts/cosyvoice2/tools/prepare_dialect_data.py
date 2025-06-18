#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='处理方言数据集JSONL文件并生成Kaldi格式数据')
    parser.add_argument('--jsonl_file', required=True, help='输入的JSONL文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化数据结构
    utt2wav = {}
    utt2text = {}
    utt2spk = {}
    spk2utt = {}
    
    # 读取JSONL文件
    print(f"处理文件: {args.jsonl_file}")
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 解析每一行
    for line in tqdm(lines, desc="解析JSONL"):
        item = json.loads(line.strip())
        
        # 提取关键信息
        key = item["key"]
        wav_path = item["source"]
        text = item["target"]
        text_language = item["text_language"]
        
        # 从key中提取说话人ID
        spk = key.split("_")[0]
        
        # 保存信息
        utt2wav[key] = wav_path
        utt2text[key] = f"{text_language}{text}"  # 包含语言标记
        utt2spk[key] = spk
        
        # 更新spk2utt
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(key)
    
    # 写入wav.scp文件
    with open(os.path.join(args.output_dir, 'wav.scp'), 'w', encoding='utf-8') as f:
        for key, wav_path in sorted(utt2wav.items()):
            f.write(f"{key} {wav_path}\n")
    
    # 写入text文件
    with open(os.path.join(args.output_dir, 'text'), 'w', encoding='utf-8') as f:
        for key, text in sorted(utt2text.items()):
            f.write(f"{key} {text}\n")
    
    # 写入utt2spk文件
    with open(os.path.join(args.output_dir, 'utt2spk'), 'w', encoding='utf-8') as f:
        for key, spk in sorted(utt2spk.items()):
            f.write(f"{key} {spk}\n")
    
    # 写入spk2utt文件
    with open(os.path.join(args.output_dir, 'spk2utt'), 'w', encoding='utf-8') as f:
        for spk, keys in sorted(spk2utt.items()):
            f.write(f"{spk} {' '.join(sorted(keys))}\n")
    
    # 显示处理结果
    print(f"处理完成!")
    print(f"总共处理了 {len(utt2wav)} 条记录")
    print(f"共有 {len(spk2utt)} 个说话人")
    print(f"输出文件保存在: {args.output_dir}")
    print(f"  - wav.scp: {len(utt2wav)} 条记录")
    print(f"  - text: {len(utt2text)} 条记录")
    print(f"  - utt2spk: {len(utt2spk)} 条记录")
    print(f"  - spk2utt: {len(spk2utt)} 条记录")

if __name__ == "__main__":
    main()