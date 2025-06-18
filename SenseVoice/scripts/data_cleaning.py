import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn.functional as F
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
# 替换 transformers 的 pipeline
from funasr import AutoModel as FunASRAutoModel
"""
python scripts/data_cleaning.py \
  --data_dir /home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_augmented \
  --output_dir /home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_cleaned \
  --jsonl_path augmented_data.jsonl \
  --sensevoice_path /home/chihan/workspace/SenseVoice/outputs/northeastern \
  --campplus_path /home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B/campplus.onnx \
  --asr_threshold 0.85 \
  --spk_threshold 0.75
"""
# 配置参数
ASR_SIM_THRESHOLD = 0.70  # ASR相似度阈值
SPK_SIM_THRESHOLD = 0.75  # 说话人相似度阈值
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DataCleaner:
    def __init__(self, data_dir: str, output_dir: str, sensevoice_path: str, campplus_path: str):
        """
        初始化数据清洗器
        
        Args:
            data_dir: 包含增强数据的目录
            output_dir: 清洗后数据的保存目录
            sensevoice_path: SenseVoice模型路径
            campplus_path: campplus.onnx模型路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sensevoice_path = sensevoice_path
        self.campplus_path = campplus_path
        
        # 加载模型
        print("正在加载模型...")
        self.load_models()
        
        # 统计
        self.total_files = 0
        self.accepted_files = 0
        self.rejected_asr = 0
        self.rejected_spk = 0
    
    def load_models(self):
        """加载ASR模型和说话人嵌入模型"""
        # 加载SenseVoice ASR模型
        print(f"正在加载SenseVoice模型: {self.sensevoice_path}")
        self.asr_model = FunASRAutoModel(
            model=self.sensevoice_path,
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            disable_update=True
        )
        
        # 加载本地SIMCSE模型用于计算文本相似度
        simcse_local_path = "/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/princeton-nlp/sup-simcse-bert-base-uncased"
        print(f"正在从本地加载SIMCSE模型: {simcse_local_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(simcse_local_path)
        self.simcse_model = AutoModel.from_pretrained(simcse_local_path).to(DEVICE)
        
        # 加载CosyVoice中的campplus.onnx模型
        print(f"正在加载CampPlus模型: {self.campplus_path}")
        self.spk_model = ort.InferenceSession(
            self.campplus_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
            if DEVICE == "cuda" else ['CPUExecutionProvider']
        )
        
        print("所有模型加载完成!")
    
    def get_asr_transcript(self, audio_path: str) -> str:
        """
        使用SenseVoice模型获取音频的转录文本
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录文本
        """
        try:
            # 使用FunASR的AutoModel进行推理
            result = self.asr_model.generate(input=audio_path, language="zh", use_itn=True)
            transcript = result[0]["text"] if isinstance(result, list) else result["text"]
            return transcript
        except Exception as e:
            print(f"ASR处理出错: {e}")
            return ""
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        使用SIMCSE计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # 分词
        inputs = self.tokenizer([text1, text2], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        
        # 获取嵌入
        with torch.no_grad():
            embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
        
        # 计算余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings[0], embeddings[1])
        
        return similarity.item()
    
    def get_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """使用CampPlus模型提取说话人嵌入"""
        try:
            # 加载音频
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
            
            # 提取Fbank特征 (80维)
            fbank = librosa.feature.melspectrogram(
                y=waveform, 
                sr=sample_rate,
                n_mels=80,  # 80维梅尔频谱
                n_fft=400,
                hop_length=160,
                win_length=400
            )
            
            # 转换为对数刻度
            log_fbank = np.log(fbank + 1e-6)
            
            # 归一化特征
            mean = np.mean(log_fbank, axis=1, keepdims=True)
            std = np.std(log_fbank, axis=1, keepdims=True)
            normalized_fbank = (log_fbank - mean) / (std + 1e-8)
            
            # 转置并调整维度为 [batch_size, channels, time_dim, freq_dim]
            # 或者 [batch_size, time_dim, freq_dim] 取决于模型需求
            feature = normalized_fbank.T  # [time, freq]
            feature = np.expand_dims(feature, axis=0)  # [batch, time, freq]
            
            # 确保输入数据类型正确
            feature = feature.astype(np.float32)
            
            # 获取输入节点名称
            input_name = self.spk_model.get_inputs()[0].name
            
            # 运行ONNX模型
            embedding = self.spk_model.run(None, {input_name: feature})[0]
            
            # 标准化嵌入
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding
        except Exception as e:
            print(f"提取说话人嵌入出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            return np.zeros((1, 192), dtype=np.float32)
            
    def calculate_speaker_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个说话人嵌入的相似度
        
        Args:
            emb1: 第一个说话人嵌入
            emb2: 第二个说话人嵌入
            
        Returns:
            相似度分数 (0-1)
        """
        # 确保嵌入已归一化
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # 计算余弦相似度
        similarity = np.dot(emb1_norm.flatten(), emb2_norm.flatten())
        return float(similarity)
    
    def process_file(self, item: Dict) -> Tuple[bool, Dict, str]:
        """
        处理单个数据条目
        
        Args:
            item: 数据条目
            
        Returns:
            (是否接受, 带有相似度分数的条目, 拒绝原因)
        """
        audio_path = item["source"]
        reference_text = item["target"]
        reference_audio = item.get("reference_audio", "")  # 参考音频路径
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在 {audio_path}")
            return False, item, "file_not_found"
            
        # 获取ASR转录
        asr_transcript = self.get_asr_transcript(audio_path)
        
        # 计算ASR相似度
        asr_similarity = self.calculate_text_similarity(reference_text, asr_transcript)
        
        # 如果ASR相似度不足，直接拒绝
        if asr_similarity < ASR_SIM_THRESHOLD:
            item["asr_similarity"] = asr_similarity
            item["asr_transcript"] = asr_transcript
            return False, item, "asr"
        
        # 如果提供了参考音频，计算说话人相似度
        if reference_audio and os.path.exists(reference_audio):
            ref_embedding = self.get_speaker_embedding(reference_audio)
            sample_embedding = self.get_speaker_embedding(audio_path)
            spk_similarity = self.calculate_speaker_similarity(sample_embedding, ref_embedding)
            
            # 保存相似度分数
            item["asr_similarity"] = asr_similarity
            item["asr_transcript"] = asr_transcript
            item["speaker_similarity"] = spk_similarity
            
            # 基于两个相似度判断是否接受
            if spk_similarity < SPK_SIM_THRESHOLD:
                return False, item, "spk"
            else:
                return True, item, "accept"
        else:
            # 如果没有参考音频，只基于ASR相似度判断
            item["asr_similarity"] = asr_similarity
            item["asr_transcript"] = asr_transcript
            return True, item, "accept"
    
    def clean_dataset(self, jsonl_path: str):
        """
        清洗数据集
        
        Args:
            jsonl_path: 数据集JSONL文件路径
        """
        input_file = self.data_dir / jsonl_path
        accept_file = self.output_dir / f"cleaned_{input_file.name}"
        reject_file = self.output_dir / f"rejected_{input_file.name}"
        
        # 读取数据
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析JSON行: {line}")
        
        print(f"加载了 {len(data)} 条数据")
        
        # 将原始数据按文本内容进行索引
        original_data_by_text = {}
        for item in data:
            source_path = item["source"]
            # 检测是否为原始数据
            if "KeSpeech_Northeastern/audio/" in source_path:
                text = item["target"]
                original_data_by_text[text] = item
        
        print(f"找到 {len(original_data_by_text)} 条不同文本的原始数据")
        
        accepted_data = []
        rejected_data = []
        
        # 处理每个增强数据条目
        for item in tqdm(data, desc="清洗数据"):
            # 跳过原始数据，只处理增强数据
            if "KeSpeech_Northeastern/audio/" in item["source"]:
                continue
                
            self.total_files += 1
            text = item["target"]
            
            # 查找对应的原始音频
            if text in original_data_by_text:
                original_item = original_data_by_text[text]
                reference_audio = original_item["source"]
                
                # 添加参考音频路径
                item["reference_audio"] = reference_audio
                is_accepted, item_with_scores, reason = self.process_file(item)
            else:
                print(f"警告: 未找到文本 '{text}' 的原始音频")
                is_accepted = False
                item_with_scores = item
                reason = "no_reference"
                
            if is_accepted:
                self.accepted_files += 1
                accepted_data.append(item_with_scores)
            else:
                if reason == "asr":
                    self.rejected_asr += 1
                elif reason == "spk":
                    self.rejected_spk += 1
                rejected_data.append(item_with_scores)
        
        # 保存清洗后的数据
        with open(accept_file, 'w', encoding='utf-8') as f:
            for item in accepted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        with open(reject_file, 'w', encoding='utf-8') as f:
            for item in rejected_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 打印统计信息
        print(f"数据清洗完成!")
        print(f"总文件数: {self.total_files}")
        print(f"接受文件数: {self.accepted_files} ({self.accepted_files/self.total_files*100:.2f}%)")
        print(f"因ASR相似度低而拒绝: {self.rejected_asr} ({self.rejected_asr/self.total_files*100:.2f}%)")
        print(f"因说话人相似度低而拒绝: {self.rejected_spk} ({self.rejected_spk/self.total_files*100:.2f}%)")
        
        # 保存清洗报告
        report = {
            "total_files": self.total_files,
            "accepted_files": self.accepted_files,
            "rejected_asr": self.rejected_asr,
            "rejected_spk": self.rejected_spk,
            "acceptance_rate": float(self.accepted_files)/float(self.total_files) if self.total_files > 0 else 0,
            "thresholds": {
                "asr_similarity": ASR_SIM_THRESHOLD,
                "speaker_similarity": SPK_SIM_THRESHOLD
            }
        }
        
        with open(self.output_dir / "cleaning_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="语音数据清洗脚本")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_augmented_2x",
                        help="包含增强数据的目录")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_2x_cleaned",
                        help="清洗后数据的保存目录")
    parser.add_argument("--jsonl_path", type=str, 
                        default="/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern_augmented_2x/combined_data.jsonl",
                        help="数据集JSONL文件名")
    parser.add_argument("--sensevoice_path", type=str, 
                        default="/home/chihan/workspace/SenseVoice/outputs/northeastern",
                        help="SenseVoice模型路径")
    parser.add_argument("--campplus_path", type=str, 
                        default="/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B/campplus.onnx",
                        help="CampPlus ONNX模型路径")
    parser.add_argument("--asr_threshold", type=float, default=0.5, help="ASR相似度阈值")
    parser.add_argument("--spk_threshold", type=float, default=0.75, help="说话人相似度阈值")
    
    args = parser.parse_args()
    
    # 更新全局阈值
    ASR_SIM_THRESHOLD = args.asr_threshold
    SPK_SIM_THRESHOLD = args.spk_threshold
    
    # 创建清洗器并执行清洗
    cleaner = DataCleaner(
        args.data_dir, 
        args.output_dir,
        args.sensevoice_path,
        args.campplus_path
    )
    cleaner.clean_dataset(args.jsonl_path)