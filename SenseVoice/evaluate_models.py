import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
from funasr import AutoModel
import logging
import time
import time
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path_or_name):
    """加载模型"""
    logging.info(f"正在加载模型: {model_path_or_name}")
    
    model = AutoModel(
        model=model_path_or_name,  # 无论本地还是远程，都直接使用模型路径或名称
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
        disable_update=True
    )
    
    return model

def clean_text(text):
    """清理文本，移除特殊标记"""
    # 移除情感和事件标签
    special_tokens = ["<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", 
                      "<|FEARFUL|>", "<|DISGUSTED|>", "<|SURPRISED|>", 
                      "<|Speech|>", "<|woitn|>", "<|zh|>"]
    
    # 添加更多需要移除的特殊标记
    additional_tokens = ["<|withitn|>", "<|EMO_UNKNOWN|>", "<|withoitn|>",
                         "<|itn|>", "<|symbol|>", "<|map|>", "<|end|>", "<|punct|>"]
    special_tokens.extend(additional_tokens)
    
    # 移除所有特殊标记
    for token in special_tokens:
        text = text.replace(token, "")
    
    # 移除标点符号（可选，取决于您的评估需求）
    # import re
    # text = re.sub(r'[^\w\s]', '', text)
    
    # 移除多余空格
    text = " ".join(text.split())
    
    # 移除句尾的标点符号（如句号、问号等）
    text = text.rstrip('。.?!？！，,、；;：:')
    
    return text.strip()

def evaluate_model(model, test_data_path, output_file=None, max_samples=None, batch_size=8):
    """使用批处理评估模型性能"""
    results = []
    all_refs = []
    all_hyps = []
    
    # 加载测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    if max_samples and max_samples < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
    
    total_samples = len(test_data)
    logging.info(f"开始评估 {total_samples} 条测试数据，使用批大小: {batch_size}")
    
    # 将数据分成批次
    num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整
    
    pbar = tqdm(
        total=total_samples,
        desc="评估进度",
        ncols=100,
        mininterval=0.5,
        maxinterval=2.0,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    
    total_time = 0
    success_count = 0
    error_count = 0
    
    # 处理每个批次
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_items = test_data[start_idx:end_idx]
        
        # 收集该批次的音频路径和引用文本
        batch_audio_paths = [item["source"] for item in batch_items]
        batch_references = [clean_text(item["target"]) for item in batch_items]
        
        try:
            start_time = time.time()
            
            # 批处理推理
            batch_results = model.generate(input=batch_audio_paths, language="zh", use_itn=True)
            batch_process_time = time.time() - start_time
            
            # 处理每个样本的结果
            for i, (item, ref, result) in enumerate(zip(batch_items, batch_references, batch_results)):
                try:
                    hypothesis = clean_text(result["text"])
                    
                    # 计算单样本的处理时间 (大致平均)
                    sample_process_time = batch_process_time / len(batch_results)
                    total_time += sample_process_time
                    
                    # 计算单样本的CER
                    sample_cer = cer(ref, hypothesis)
                    sample_wer = wer(ref, hypothesis)  # 添加WER计算
                    
                    result_item = {
                        "audio": item["source"],
                        "reference": ref,
                        "hypothesis": hypothesis,
                        "cer": sample_cer,
                        "wer": sample_wer,  # 添加WER记录
                        "process_time": sample_process_time
                    }
                    
                    results.append(result_item)
                    all_refs.append(ref)
                    all_hyps.append(hypothesis)
                    
                    success_count += 1
                    
                    # 更新进度条信息
                    if success_count > 0:
                        avg_time = total_time / success_count
                        pbar.set_postfix({
                            "平均RTF": f"{avg_time:.3f}s",
                            "当前CER": f"{sample_cer:.4f}",
                            "当前WER": f"{sample_wer:.4f}",  # 添加WER显示
                            "批大小": batch_size
                        })
                        
                except Exception as e:
                    logging.error(f"处理批次结果时出错: {item['source']}, 错误: {str(e)}")
                    error_count += 1
                
                # 更新进度条
                pbar.update(1)
                
        except Exception as e:
            logging.error(f"处理批次时出错, 批次 {batch_idx+1}/{num_batches}, 错误: {str(e)}")
            # 对于这个批次的所有样本，标记为失败
            error_count += end_idx - start_idx
            pbar.update(end_idx - start_idx)
    
    pbar.close()
    
    # 计算整体性能指标
    if all_refs and all_hyps:
        overall_cer = cer(all_refs, all_hyps)
        overall_wer = wer(all_refs, all_hyps)
        
        logging.info(f"整体字错率(CER): {overall_cer:.4f}")
        logging.info(f"整体词错率(WER): {overall_wer:.4f}")
    else:
        logging.error("没有成功处理任何样本，无法计算性能指标")
        overall_cer = overall_wer = 1.0  # 设置为最大错误率
    
    # 添加处理统计
    logging.info(f"处理完成: 总样本 {total_samples}, 成功 {success_count}, 失败 {error_count}")
    if success_count > 0:
        logging.info(f"平均处理时间: {total_time/success_count:.3f}秒/样本")
        logging.info(f"使用批大小: {batch_size}, 总批次: {num_batches}")
    
    # 保存详细结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "overall_metrics": {
                    "cer": overall_cer,
                    "wer": overall_wer,
                    "avg_process_time": total_time / success_count if success_count > 0 else None,
                    "success_rate": success_count / total_samples if total_samples > 0 else 0,
                    "batch_size": batch_size
                },
                "samples": results
            }, f, ensure_ascii=False, indent=2)
    
    return overall_cer, overall_wer, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估SenseVoice模型性能')
    parser.add_argument('--original_model', default="iic/SenseVoiceSmall", 
                        help='原始模型名称或路径')
    parser.add_argument('--finetuned_model', default="/home/chihan/workspace/SenseVoice/outputs/aug_new_0.1x_Northeastern_finetune", 
                        help='微调模型目录路径')
    parser.add_argument('--test_data', default='/home/chihan/workspace/SenseVoice/data/KeSpeech_Northeastern/dialect_test.jsonl', 
                        help='测试数据JSONL文件路径')
    parser.add_argument('--output_dir', default='./eval_results/aug_0.1x_northeastern', 
                        help='评估结果输出目录')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='最大评估样本数，为None时评估全部')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小，更大的值可能会加快处理速度（取决于GPU内存）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='并行工作进程数（仅在使用多进程时有效）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time_all = time.time()
    
    logging.info("===== 开始评估流程 =====")
    
    # 评估原始模型
    logging.info("\n1. 评估原始模型...")
    start_time = time.time()
    original_model = load_model(args.original_model)
    original_cer, original_wer, original_results = evaluate_model(
        original_model, 
        args.test_data, 
        os.path.join(args.output_dir, "original_model_results.json"),
        args.max_samples,
        args.batch_size  # 添加批大小参数
    )
    logging.info(f"原始模型评估完成，耗时: {(time.time() - start_time):.2f}秒")
    
    # 评估微调模型
    logging.info("\n2. 评估微调模型...")
    start_time = time.time()
    finetuned_model = load_model(args.finetuned_model)
    finetuned_cer, finetuned_wer, finetuned_results = evaluate_model(
        finetuned_model, 
        args.test_data, 
        os.path.join(args.output_dir, "finetuned_model_results.json"),
        args.max_samples,
        args.batch_size  # 添加批处理大小参数，确保与原始模型评估一致
    )
    logging.info(f"微调模型评估完成，耗时: {(time.time() - start_time):.2f}秒")
    
    # 生成比较报告
    cer_improvement = (original_cer - finetuned_cer) / original_cer * 100
    wer_improvement = (original_wer - finetuned_wer) / original_wer * 100
    
    comparison = {
        "original_model": {
            "name": args.original_model,
            "cer": original_cer,
            "wer": original_wer
        },
        "finetuned_model": {
            "name": args.finetuned_model,
            "cer": finetuned_cer,
            "wer": finetuned_wer
        },
        "improvement": {
            "cer_absolute": original_cer - finetuned_cer,
            "wer_absolute": original_wer - finetuned_wer,
            "cer_relative": f"{cer_improvement:.2f}%",
            "wer_relative": f"{wer_improvement:.2f}%"
        }
    }
    
    # 保存比较报告
    with open(os.path.join(args.output_dir, "comparison_report.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 打印比较结果
    logging.info("\n========== 模型比较结果 ==========")
    logging.info(f"原始模型 ({args.original_model}):")
    logging.info(f"  CER: {original_cer:.4f}")
    logging.info(f"  WER: {original_wer:.4f}")
    logging.info(f"微调模型 ({args.finetuned_model}):")
    logging.info(f"  CER: {finetuned_cer:.4f}")
    logging.info(f"  WER: {finetuned_wer:.4f}")
    logging.info(f"性能提升:")
    logging.info(f"  CER绝对减少: {original_cer - finetuned_cer:.4f}")
    logging.info(f"  WER绝对减少: {original_wer - finetuned_wer:.4f}")
    logging.info(f"  CER相对提升: {cer_improvement:.2f}%")
    logging.info(f"  WER相对提升: {wer_improvement:.2f}%")
    logging.info("==================================")
    
    total_time = time.time() - start_time_all
    logging.info(f"\n评估流程完成，总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")