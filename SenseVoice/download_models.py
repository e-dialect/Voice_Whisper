import os
import sys
import argparse
import requests
import shutil
import zipfile
import tarfile
from tqdm import tqdm
import torch

# 设置基本路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache", "models")
UVR5_WEIGHTS_DIR = os.path.join(BASE_DIR, "tools", "uvr5", "uvr5_weights")
COSYVOICE_DIR = os.path.join(BASE_DIR, "CosyVoice", "pretrained_models")

# 创建必要的目录
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(UVR5_WEIGHTS_DIR, exist_ok=True)
os.makedirs(COSYVOICE_DIR, exist_ok=True)

# 定义需要下载的模型列表
MODELSCOPE_MODELS = {
    # 基础ASR模型
    "iic/SenseVoiceSmall": {
        "description": "SenseVoice标准普通话模型",
        "size": "1.2GB"
    },
    # 辅助模型
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": {
        "description": "时间戳和说话人分离模型",
        "size": "1.0GB"
    },
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch": {
        "description": "语音活动检测(VAD)模型",
        "size": "5MB"
    },
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch": {
        "description": "标点恢复模型",
        "size": "500MB"
    },
    "iic/speech_campplus_sv_zh-cn_16k-common": {
        "description": "说话人识别模型",
        "size": "80MB"
    }
}

# 降噪模型下载信息
UVR5_MODELS = {
    "HP2": {
        "url": "https://huggingface.co/bryanchin/uvr5/resolve/main/HP2_all_vocals.pth",
        "description": "HP2 all vocals 降噪模型",
        "size": "140MB"
    },
    "HP5": {
        "url": "https://huggingface.co/bryanchin/uvr5/resolve/main/HP5_only_main_vocal.pth",
        "description": "HP5 only main vocal 降噪模型",
        "size": "140MB"
    },
    "VR-DeEcho": {
        "url": "https://huggingface.co/bryanchin/uvr5/resolve/main/VR-DeEchoNormal.pth",
        "description": "VR DeEcho Normal 降噪模型",
        "size": "140MB"
    },
    "VR-DeEcho-Aggressive": {
        "url": "https://huggingface.co/bryanchin/uvr5/resolve/main/VR-DeEchoAggressive.pth",
        "description": "VR DeEcho Aggressive 降噪模型",
        "size": "140MB"
    },
    "VR-DeEcho-DeReverb": {
        "url": "https://huggingface.co/bryanchin/uvr5/resolve/main/VR-DeEchoDeReverb.pth",
        "description": "VR DeEcho DeReverb 降噪模型",
        "size": "140MB"
    }
}

# CosyVoice模型下载信息
COSYVOICE_MODELS = {
    "CosyVoice2-0.5B": {
        "url": "https://huggingface.co/thuhcsi/CosyVoice2-0.5B/resolve/main/CosyVoice2-0.5B.zip",
        "description": "CosyVoice2-0.5B 声音克隆模型",
        "size": "2.5GB"
    }
}

# 方言微调模型信息
DIALECT_MODELS = {
    "Jiang-Huai": {
        "description": "江淮方言模型 (需从方言训练目录复制)",
        "size": "1.2GB",
        "path": os.path.join(BASE_DIR, "outputs", "Jiang-Huai")
    },
    "northeastern": {
        "description": "东北方言模型 (需从方言训练目录复制)",
        "size": "1.2GB",
        "path": os.path.join(BASE_DIR, "outputs", "northeastern")
    },
    "south_western": {
        "description": "西南方言模型 (需从方言训练目录复制)",
        "size": "1.2GB",
        "path": os.path.join(BASE_DIR, "outputs", "south_western")
    }
}

def download_file(url, destination, description=None):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        desc = description if description else os.path.basename(destination)
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            print(f"下载 {description} 失败，文件可能不完整")
            return False
        return True
    except Exception as e:
        print(f"下载 {description} 时出错: {e}")
        return False

def extract_archive(archive_path, extract_dir):
    """解压缩文件"""
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"解压缩 {archive_path} 时出错: {e}")
        return False

def download_modelscope_models():
    """下载ModelScope模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        
        for model_id, info in MODELSCOPE_MODELS.items():
            model_path = os.path.join(MODEL_CACHE_DIR, *model_id.split('/'))
            
            if os.path.exists(model_path):
                print(f"✓ {info['description']} 已存在，跳过下载")
                continue
                
            print(f"下载 {info['description']} ({info['size']})...")
            try:
                snapshot_download(model_id, cache_dir=MODEL_CACHE_DIR)
                print(f"✓ {info['description']} 下载完成")
            except Exception as e:
                print(f"× {info['description']} 下载失败: {e}")
    except ImportError:
        print("请先安装ModelScope: pip install modelscope")
        return False
    return True

def download_uvr5_models():
    """下载UVR5降噪模型"""
    success = True
    for model_name, info in UVR5_MODELS.items():
        model_path = os.path.join(UVR5_WEIGHTS_DIR, f"{model_name}.pth")
        if model_name == "HP2":
            model_path = os.path.join(UVR5_WEIGHTS_DIR, "HP2_all_vocals.pth")
        elif model_name == "HP5":
            model_path = os.path.join(UVR5_WEIGHTS_DIR, "HP5_only_main_vocal.pth")
        elif model_name == "VR-DeEcho":
            model_path = os.path.join(UVR5_WEIGHTS_DIR, "VR-DeEchoNormal.pth")
        elif model_name == "VR-DeEcho-Aggressive":
            model_path = os.path.join(UVR5_WEIGHTS_DIR, "VR-DeEchoAggressive.pth")
        elif model_name == "VR-DeEcho-DeReverb":
            model_path = os.path.join(UVR5_WEIGHTS_DIR, "VR-DeEchoDeReverb.pth")
            
        if os.path.exists(model_path):
            print(f"✓ {info['description']} 已存在，跳过下载")
            continue
            
        print(f"下载 {info['description']} ({info['size']})...")
        success = download_file(info['url'], model_path, info['description']) and success
    
    # 特殊处理onnx_dereverb_By_FoxJoy模型
    onnx_dir = os.path.join(UVR5_WEIGHTS_DIR, "onnx_dereverb_By_FoxJoy")
    onnx_file = os.path.join(onnx_dir, "vocals.onnx")
    if not os.path.exists(onnx_file):
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_url = "https://huggingface.co/bryanchin/uvr5/resolve/main/onnx_dereverb_By_FoxJoy/vocals.onnx"
        print(f"下载 onnx_dereverb_By_FoxJoy 模型...")
        success = download_file(onnx_url, onnx_file, "onnx_dereverb_By_FoxJoy") and success
    else:
        print(f"✓ onnx_dereverb_By_FoxJoy 模型已存在，跳过下载")
        
    return success

def download_cosyvoice_models():
    """下载CosyVoice模型"""
    success = True
    for model_name, info in COSYVOICE_MODELS.items():
        model_dir = os.path.join(COSYVOICE_DIR, model_name)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # 检查是否包含关键文件
            if os.path.exists(os.path.join(model_dir, "llm.pt")) and os.path.exists(os.path.join(model_dir, "flow.pt")):
                print(f"✓ {info['description']} 已存在，跳过下载")
                continue
        
        # 创建临时下载目录
        zip_path = os.path.join(COSYVOICE_DIR, f"{model_name}.zip")
        print(f"下载 {info['description']} ({info['size']})...")
        
        if download_file(info['url'], zip_path, info['description']):
            print(f"解压 {model_name}...")
            if extract_archive(zip_path, COSYVOICE_DIR):
                print(f"✓ {model_name} 解压完成")
                # 清理zip文件
                os.remove(zip_path)
            else:
                success = False
        else:
            success = False
    
    return success

def check_dialect_models():
    """检查方言微调模型是否存在"""
    for model_name, info in DIALECT_MODELS.items():
        if os.path.exists(info['path']):
            print(f"✓ {info['description']} 已存在于 {info['path']}")
        else:
            print(f"× {info['description']} 不存在")
            print(f"  请将微调后的方言模型放置在 {info['path']} 目录中")

def main():
    parser = argparse.ArgumentParser(description="下载SenseVoice项目所需的预训练模型")
    parser.add_argument("--asr-only", action="store_true", help="仅下载ASR相关模型")
    parser.add_argument("--uvr-only", action="store_true", help="仅下载UVR5降噪模型")
    parser.add_argument("--cosyvoice-only", action="store_true", help="仅下载CosyVoice模型")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    args = parser.parse_args()
    
    # 默认行为与--all相同
    if not (args.asr_only or args.uvr_only or args.cosyvoice_only):
        args.all = True
    
    print("SenseVoice 预训练模型下载工具")
    print("=" * 50)
    
    # 检查可用的GPU设备
    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个CUDA设备")
        print(f"当前CUDA版本: {torch.version.cuda}")
    else:
        print("未检测到CUDA设备，模型将在CPU上运行")
    
    print("=" * 50)
    
    # 下载模型
    if args.all or args.asr_only:
        print("\n开始下载ASR相关模型...")
        download_modelscope_models()
        print("\n检查方言微调模型...")
        check_dialect_models()
    
    if args.all or args.uvr_only:
        print("\n开始下载UVR5降噪模型...")
        download_uvr5_models()
    
    if args.all or args.cosyvoice_only:
        print("\n开始下载CosyVoice模型...")
        download_cosyvoice_models()
    
    print("\n所有请求的模型下载完成")
    print("如果某些模型下载失败，请尝试再次运行此脚本或手动下载")

if __name__ == "__main__":
    main()
