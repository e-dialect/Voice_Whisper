import os
import threading
import queue
import uuid
import time
from datetime import timedelta, datetime
from pydub import AudioSegment
import ffmpeg
from funasr import AutoModel
from flask import Flask, render_template, request, jsonify, send_from_directory
import werkzeug.utils
import re
import sys

sys.path.append('/home/chihan/workspace/SenseVoice/tools')
sys.path.append('/home/chihan/workspace/SenseVoice/tools/uvr5')

# 添加CosyVoice路径
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice')
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice/third_party/Matcha-TTS')

# 导入CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    cosyvoice_available = True
except ImportError:
    print("警告: CosyVoice模块未安装，声音克隆功能将不可用")
    cosyvoice_available = False

# 在全局变量部分添加
cosyvoice_model = None
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局队列
spk_txt_queue = queue.Queue()
result_queue = queue.Queue()
audio_concat_queue = queue.Queue()

# 支持的音视频格式
support_audio_format = ['.mp3', '.m4a', '.aac', '.ogg', '.wav', '.flac', '.wma', '.aif']
support_video_format = ['.mp4', '.avi', '.mov', '.mkv']

# 模型配置部分
home_directory = os.path.expanduser("~")

# ASR模型配置
models = {
    "timestamp": {  # 添加时间戳专用模型
        "path": "/home/chihan/workspace/SenseVoice/model_cache/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "None",  # 本地模型不需要revision
        "name": "时间戳模型(不可选)"
    },
    "sensevoice": {
        "path": "/home/chihan/workspace/SenseVoice/model_cache/models/iic/SenseVoiceSmall",
        "revision": "None",  # 本地模型不需要revision
        "name": "SenseVoice标准普通话"
    },
    "jiang_huai": {
        "path": "/home/chihan/workspace/SenseVoice/outputs/Jiang-Huai",
        "revision": None,  # 本地模型不需要revision
        "name": "江淮方言"
    },
    "northeastern": {
        "path": "/home/chihan/workspace/SenseVoice/outputs/northeastern",
        "revision": None,  # 本地模型不需要revision
        "name": "东北方言"
    },
    "southwestern": {
        "path": "/home/chihan/workspace/SenseVoice/outputs/south_western",
        "revision": None,  # 本地模型不需要revision
        "name": "西南方言"
    }
}

# 默认使用SenseVoice模型
default_model = "sensevoice"

# 保留其他必要的预训练模型 - 直接使用ModelScope格式的路径
vad_model_path = "/home/chihan/workspace/SenseVoice/model_cache/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
vad_model_revision = "None"  # 本地模型不需要revision
punc_model_path = "/home/chihan/workspace/SenseVoice/model_cache/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
punc_model_revision = "None"  # 本地模型不需要revision
spk_model_path = "/home/chihan/workspace/SenseVoice/model_cache/models/iic/speech_campplus_sv_zh-cn_16k-common"
spk_model_revision = "None"  # 本地模型不需要revision

# 根据系统配置选择设备
try:
    import torch
    ngpu = 1 if torch.cuda.is_available() else 0
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
except:
    ngpu = 0
    device = "cpu"
    
ncpu = os.cpu_count() or 4



# 降噪相关模块导入
try:
    from tools.my_utils import load_audio, clean_path
    # 正确导入UVR5模块中的类，而不是函数
    from tools.uvr5.vr import AudioPre, AudioPreDeEcho
    from tools.uvr5.mdxnet import MDXNetDereverb
    uvr_available = True
    print("降噪模块加载成功")
except ImportError as e:
    print(f"警告: 降噪模块加载失败: {e}")
    uvr_available = False
# 在模型配置部分添加uvr5模型配置
uvr5_models = {
    "HP2": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/HP2_all_vocals.pth",
    "HP5": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/HP5_only_main_vocal.pth",
    "VR-DeEcho": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/VR-DeEchoNormal.pth",
    "VR-DeEcho-Aggressive": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/VR-DeEchoAggressive.pth",
    "VR-DeEcho-DeReverb": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/VR-DeEchoDeReverb.pth",
    "onnx_dereverb_By_FoxJoy": "/home/chihan/workspace/SenseVoice/tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
}

# 修改process_denoise函数以匹配webui.py的uvr函数实现
def process_denoise(input_file, model_name="HP5", agg_level=10):
    """
    对音频文件进行降噪处理
    
    参数:
    input_file: 输入音频文件路径
    model_name: 降噪模型名称
    agg_level: 降噪激进程度 (0-20)
    
    返回:
    成功时返回处理后的音频路径，失败时返回None
    """
    if not uvr_available:
        print("降噪模块不可用")
        return None
    
    # 改进重复处理检测 - 更精确的判断
    if 'denoised' in input_file or 'vocal_' in input_file:
        print(f"文件已经过降噪处理，跳过: {input_file}")
        return input_file
        
    try:
        # 确保输入文件存在
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return None

        # 确保模型存在
        if model_name not in uvr5_models:
            print(f"找不到指定的降噪模型: {model_name}")
            model_name = "HP5"  # 使用默认模型
            
        # 创建输出目录
        output_vocal_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'denoised', 'vocal')
        output_inst_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'denoised', 'inst')
        os.makedirs(output_vocal_dir, exist_ok=True)
        os.makedirs(output_inst_dir, exist_ok=True)
        
        # 提前定义确定的输出文件名，避免后续猜测
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # 避免长文件名
        if len(base_name) > 20:
            base_name = base_name[:20]
            
        # 预先确定输出文件名 - 不使用UVR5的自动命名
        output_filename = f"denoised_{base_name}_{timestamp}.wav"
        output_vocal_path = os.path.join(output_vocal_dir, output_filename)
        
        # 选择正确的模型处理类
        is_hp3 = "HP3" in model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_half = torch.cuda.is_available() # 如果用GPU则使用半精度
        
        # 音频格式转换
        need_reformat = True
        inp_path = input_file
        tmp_path = None
        
        try:
            # 检查是否需要重新格式化
            info = ffmpeg.probe(inp_path, cmd="ffprobe")
            if (
                info["streams"][0]["channels"] == 2
                and info["streams"][0]["sample_rate"] == "44100"
            ):
                need_reformat = False
        except Exception as e:
            print(f"检查音频格式失败: {e}")
            need_reformat = True
            
        # 如果需要重新格式化，转换为标准格式
        if need_reformat:
            print("音频需要重新格式化为标准格式")
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base_name}_{timestamp}.wav")
            try:
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path
                print(f"格式化音频完成: {tmp_path}")
            except Exception as format_err:
                print(f"格式化音频失败: {format_err}")
                return None
        
        # 创建处理器并执行音频处理
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg=int(agg_level),
                    model_path=uvr5_models[model_name],
                    device=device,
                    is_half=is_half,
                )
            
            # 执行音频处理 - 使用UVR5的处理
            format0 = "wav"
            pre_fun._path_audio_(inp_path, output_inst_dir, output_vocal_dir, format0, is_hp3)
            
            # 清理资源
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
            except Exception as clean_err:
                print(f"清理模型资源时出错: {clean_err}")
                
            # 释放CUDA内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 查找UVR5处理后的最新文件 - 用绝对时间而不是文件名
            newest_file = None
            newest_time = 0
            
            for file in os.listdir(output_vocal_dir):
                file_path = os.path.join(output_vocal_dir, file)
                # 获取文件修改时间
                file_time = os.path.getmtime(file_path)
                
                # 选择最新创建的文件
                if file_time > newest_time:
                    newest_time = file_time
                    newest_file = file_path
            
            # 如果找到文件，将其重命名为我们预定义的文件名
            if newest_file and os.path.exists(newest_file):
                try:
                    # 使用预定义的名称，确保一致性
                    os.rename(newest_file, output_vocal_path)
                    print(f"将降噪输出文件重命名为: {output_vocal_path}")
                    vocal_file = output_vocal_path
                except Exception as rename_err:
                    print(f"重命名失败，使用原始文件: {rename_err}")
                    vocal_file = newest_file
            else:
                print("找不到降噪处理输出文件")
                return None
                
            # 返回处理后的文件路径
            if vocal_file and os.path.exists(vocal_file):
                print(f"降噪成功: {vocal_file}")
                return vocal_file
            else:
                print(f"找不到输出的人声文件")
                return None
                
        except Exception as process_err:
            print(f"音频处理异常: {process_err}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"降噪处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 删除临时文件
        if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"已删除临时文件: {tmp_path}")
            except:
                pass
# 降噪API端点
@app.route('/api/denoise', methods=['POST'])
def denoise_audio():
    """降噪API"""
    if not uvr_available:
        return jsonify({"status": "error", "message": "降噪模块不可用"})
        
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "未选择文件"})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "未选择文件"})
            
        # 获取参数
        model_name = request.form.get('model', "HP5")
        try:
            agg_level = int(request.form.get('agg_level', 10))
        except ValueError:
            agg_level = 10  # 默认值
            
        # 保存上传的文件
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"denoise_{int(time.time())}_{filename}")
        file.save(file_path)
            
        # 执行降噪处理
        denoised_file = process_denoise(file_path, model_name, agg_level)
        
        # 检查处理结果
        if denoised_file:
            # 生成相对URL路径
            relative_path = os.path.relpath(denoised_file, app.config['OUTPUT_FOLDER'])
            audio_url = f"/outputs/{relative_path}"
            
            return jsonify({
                "status": "success",
                "message": "降噪处理成功",
                "original_file": file_path,
                "denoised_file": denoised_file,
                "audio_url": audio_url
            })
        else:
            return jsonify({
                "status": "error",
                "message": "降噪处理失败"
            })
            
    except Exception as e:
        print(f"降噪API异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"降噪处理失败: {str(e)}"
        })
# 模型字典，用于缓存已加载的模型
loaded_models = {}
# 添加到辅助函数部分
def clean_emoji_text(text):
    """移除文本中的所有表情符号"""
    if not text:
        return ""
        
    # 移除所有emo_set和event_set中的表情符号
    clean_text = text
    for emoji in list(emo_set) + list(event_set):
        clean_text = clean_text.replace(emoji, "")
        
    # 移除多余空格
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text
# 添加情感识别格式化函数
def format_emotion_text(text):
    """将SenseVoice情感识别结果格式化为表情和纯文本"""
    if text is None:
        print("警告: format_emotion_text收到None输入")
        return {
            "text": "",
            "emotion": "😐",
            "events": "",
            "formatted_text": ""
        }
        
    # 保存原始文本以便调试
    original = text
    clean_text = text
    
    try:
        # 移除所有特殊标记并替换为表情
        for tag, emoji in emoji_dict.items():
            if tag in clean_text:
                clean_text = clean_text.replace(tag, emoji)
        
        # 收集所有情感（不是只保留一个）
        found_emotions = []
        for emoji in emo_set:
            if emoji in clean_text:
                found_emotions.append(emoji)
                clean_text = clean_text.replace(emoji, "")
        
        # 定义情感优先级 (从高到低)
        emotion_priority = ["😡", "😮", "😰", "😔", "😊", "🤢", "😐"]
        
        # 根据优先级选择主要情感
        main_emotion = "😐"  # 默认中性
        if found_emotions:
            # 找出优先级最高的情感
            for priority_emotion in emotion_priority:
                if priority_emotion in found_emotions:
                    main_emotion = priority_emotion
                    break
        
        # 收集所有事件标记
        event_emojis = []
        for emoji in event_set:
            if emoji in clean_text:
                event_emojis.append(emoji)
                clean_text = clean_text.replace(emoji, "")
        
        # 清理文本并组合
        result = clean_text.strip()
        prefix = "".join(event_emojis)
        
        print(f"情感识别成功: 情感={main_emotion}, 事件={prefix}, 原始文本长度={len(original)}")
        if len(found_emotions) > 1:
            print(f"检测到多种情感: {found_emotions}, 选择 {main_emotion} 作为主要情感")
        
        return {
            "text": result,
            "emotion": main_emotion,
            "events": prefix,
            "formatted_text": f"{prefix} {result} {main_emotion}".strip(),
            "all_emotions": found_emotions  # 添加所有检测到的情感，便于高级分析
        }
        
    except Exception as e:
        print(f"格式化情感文本发生错误: {e}")
        return {
            "text": original,
            "emotion": "😐",
            "events": "",
            "formatted_text": original
        }
# 添加对单个音频片段的情感识别函数
def recognize_emotion(audio_file):
    """使用SenseVoice识别音频的情感"""
    try:
        # 检查情感识别模型是否已初始化
        if 'emotion_model' not in globals() or emotion_model is None:
            print("警告: 情感识别模型未加载")
            return {
                "text": "",
                "emotion": "😐",
                "events": "",
                "formatted_text": ""
            }
            
        # 检查文件是否存在
        if not os.path.exists(audio_file):
            print(f"文件不存在: {audio_file}")
            return {
                "text": "",
                "emotion": "😐",
                "events": "",
                "formatted_text": ""
            }
            
        # 检查文件大小
        file_size = os.path.getsize(audio_file)
        if file_size < 100:  # 文件太小，可能是空文件或损坏
            print(f"文件太小或为空: {audio_file}, 大小: {file_size} 字节")
            return {
                "text": "",
                "emotion": "😐", 
                "events": "",
                "formatted_text": ""
            }
            
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        print(f"情感识别: 正在处理音频文件 {audio_file}, 大小: {len(audio_bytes)} 字节")
        
        # 调用情感识别模型
        result = emotion_model.generate(
            input=audio_bytes, 
            cache={},
            language="zh", 
            use_itn=True,
            batch_size_s=60, 
            batch_size=1,
            merge_vad=True,
            emotion=True,           # 开启情感识别
            return_raw_text=True    # 返回带标记的原始文本
        )
        print(f"情感识别结果: {result}")        
        # 添加强大的检查
        if result is None or len(result) == 0:
            print("情感识别: 模型返回空结果")
            return {
                "text": "",
                "emotion": "😐",
                "events": "",
                "formatted_text": ""
            }
        
        # 尝试获取文本结果
        if not isinstance(result[0], dict) or "text" not in result[0]:
            print(f"情感识别: 意外的结果格式 {type(result[0])}")
            print(f"情感识别结果: {result}")
            return {
                "text": "",
                "emotion": "😐",
                "events": "",
                "formatted_text": ""
            }
        
        # 获取文本并处理
        text = result[0]["text"]
        if text is None or not isinstance(text, str):
            print(f"情感识别: 文本不是字符串 {type(text)}")
            return {
                "text": "",
                "emotion": "😐",
                "events": "",
                "formatted_text": ""
            }
        
        # 安全地格式化情感文本
        try:
            emotion_info = format_emotion_text(text)
            return emotion_info
        except Exception as e:
            print(f"格式化情感文本错误: {e}")
            return {
                "text": text,  # 至少返回原始文本
                "emotion": "😐",
                "events": "",
                "formatted_text": text
            }
            
    except Exception as e:
        print(f"情感识别错误: {str(e)}")
        return {
            "text": "",
            "emotion": "😐",
            "events": "",
            "formatted_text": ""
        }
def load_model(model_key):
    """根据模型键加载相应模型，只加载ASR模型，共用辅助模型"""
    if model_key not in loaded_models:
        print(f"正在加载模型 {models[model_key]['name']}...")
        
        # 检查模型路径是否存在
        model_path = models[model_key]['path']
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在: {model_path}")
            print(f"请确保已下载所有必要的模型文件")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 只使用本地模型
        loaded_models[model_key] = AutoModel(
            model=model_path,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
            local_files_only=True,  # 添加此参数
            provider="local"         # 指定使用本地提供者
        )
        print(f"模型 {models[model_key]['name']} 加载完成")
    
    return loaded_models[model_key]
# 创建全局时间戳模型，避免重复加载
timestamp_model = None

# 在模型配置部分添加
emotion_model_path = "/home/chihan/workspace/SenseVoice/model_cache/models/iic/SenseVoiceSmall"  # SenseVoice模型用于情感识别
emotion_model_revision = "None"  # 本地模型不需要revision

# 添加表情映射字典
emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "😐",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎵",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😄",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
}

# 完整的emoji字典
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "😐",
    "<|BGM|>": "🎵",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😄",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "🎤",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

# 情感集合和事件集合，用于后处理
emo_set = {"😊", "😔", "😡", "😐", "😰", "🤢", "😮"}
event_set = {"🎵", "👏", "😄", "😭", "🤧", "😷", "🎤"}
def initialize_app():
    """初始化应用并加载所有需要的模型"""
    global model, timestamp_model, emotion_model, cosyvoice_model
    
    # 加载CosyVoice模型用于声音克隆
    print("正在加载CosyVoice声音克隆模型...")
    try:
        if cosyvoice_available:
            cosyvoice_model = CosyVoice2(
                '/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                load_jit=False, 
                load_trt=False, 
                fp16=False
            )
            print("CosyVoice声音克隆模型加载成功")
        else:
            cosyvoice_model = None
            print("CosyVoice模块未安装，声音克隆功能不可用")
    except Exception as e:
        cosyvoice_model = None
        print(f"CosyVoice声音克隆模型加载失败: {e}")
    print("正在加载时间戳和说话人分离模型...")
     # 加载时间戳模型（只加载一次）
    timestamp_model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    model_revision="master",
    vad_model=vad_model_path,
    vad_model_revision=vad_model_revision,
    punc_model=punc_model_path,
    punc_model_revision=punc_model_revision,
    spk_model=spk_model_path,
    spk_model_revision=spk_model_revision,
    ngpu=ngpu,
    ncpu=ncpu,
    device=device,
    spk_threshold=0.2,       # 降低说话人区分阈值（默认是0.5)
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
    add_punc=True  # 确保此参数设置为True
)
    print("时间戳和说话人分离模型加载完成")
    # 添加时间戳模型测试代码
    print("时间戳模型初始化完成，开始测试...")
    test_audio = b"\x00\x00" * 16000  # 1秒静音
    test_timestamp_res = timestamp_model.generate(
        input=test_audio, 
        batch_size_s=60,
        batch_size=1,  # 确保包含这个参数
        is_final=True, 
        sentence_timestamp=True,
        speaker_change=True,
    add_punc=True  # 确保此参数设置为True
    )
    print(f"时间戳模型测试结果: {test_timestamp_res}")
    if test_timestamp_res and len(test_timestamp_res) > 0:
        print(f"结果包含的键: {list(test_timestamp_res[0].keys())}")
        if 'text' not in test_timestamp_res[0]:
            print("警告：时间戳模型返回结果中缺少'text'字段！")
    print("时间戳和说话人分离模型加载完成")
    print(f"正在预加载默认模型 {models[default_model]['name']}...")
    model = load_model(default_model)
    print(f"默认模型 {models[default_model]['name']} 加载完成")
    
    print("正在加载情感识别模型...")
    # 加载SenseVoice情感识别模型，使用完全本地模式
    emotion_model = AutoModel(
        model=emotion_model_path,
        model_revision=emotion_model_revision,
        vad_model=vad_model_path,
        vad_model_revision=vad_model_revision,
        ngpu=ngpu,
        ncpu=ncpu,
        device=device,
        disable_pbar=True,
        disable_log=True,
        disable_update=True,
        trust_remote_code=False,  # 修改为False，避免执行远程代码
        local_files_only=True,    # 强制只使用本地文件
        provider="local"          # 指定使用本地提供者
    )
    
    # 添加更多调试信息
    print("情感模型初始化完成，开始测试...")
    
    # 测试情感模型是否正常工作
    test_result = emotion_model.generate(
        input=b"\x00\x00" * 16000,  # 1秒静音
        language="auto"
    )
    print(f"情感模型测试结果: {test_result}")
    print("情感识别模型加载成功")


# 辅助函数
def to_date(milliseconds):
    """将时间戳转换为SRT格式的时间"""
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"

def to_milliseconds(time_str):
    """将时间字符串转换为毫秒"""
    try:
        # 确保传入的是字符串
        if not isinstance(time_str, str):
            print(f"警告: 非字符串时间格式 {time_str}, 类型: {type(time_str)}")
            return 0
            
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        time_delta = time_obj - datetime(1900, 1, 1)
        milliseconds = int(time_delta.total_seconds() * 1000)
        return milliseconds
    except Exception as e:
        print(f"时间转换错误: {e}, 输入值: {time_str}")
        return 0  # 返回默认值而不是抛出异常
def improve_text_formatting(text):
    """为没有标点的文本添加基本标点符号"""
    if not text:
        return ""
    
    # 1. 按自然停顿添加逗号（通常是5-10个字符之后）
    chars = list(text)
    result = ""
    for i, char in enumerate(chars):
        result += char
        # 每8-12个字符检查是否适合添加逗号
        if i > 0 and (i+1) % 10 == 0 and i < len(chars)-1:
            # 避免在某些字符后直接加逗号
            if char not in "，。？！,.?!":
                result += "，"
    
    # 2. 在句子结尾添加句号
    if result and result[-1] not in "，。？！,.?!":
        result += "。"
    
    return result
def clean_text(text):
    """移除文本中的特殊标记但保留标点符号"""
    import re
    # 移除所有类似<|xyz|>格式的标记，但保留标点
    clean = re.sub(r'<\|[^|]*\|>', '', text)
    
    # 移除可能的多余空格但保留标点
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # 检查输出是否包含基本标点
    if not re.search(r'[，。？！,.?!]', clean):
        # 如果没有标点，可以在此添加标点恢复逻辑
        clean = improve_text_formatting(clean)
        
    return clean
# 处理音频分离
def process_audio(file_path, output_folder, model_key=default_model, split_number=10, enable_denoise=False, denoise_model="HP5", denoise_level=10):
    """
    处理音频并识别文本
    
    参数:
    file_path: 音频文件路径
    output_folder: 输出目录
    model_key: 使用的语音识别模型
    split_number: 每段文字数量
    enable_denoise: 是否启用降噪
    denoise_model: 降噪模型名称
    denoise_level: 降噪强度级别(0-20)
    """
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": "文件不存在"}
        
        # 确保选择的模型有效
        if model_key not in models:
            return {"status": "error", "message": f"无效的模型选择: {model_key}"}
        
        audio_name = os.path.splitext(os.path.basename(file_path))[0]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 检查文件格式
        if file_ext not in support_audio_format + support_video_format:
            return {"status": "error", "message": f"不支持的文件格式: {file_ext}"}
        # 优先处理降噪，如果启用了
        if enable_denoise and uvr_available:
            print(f"对音频进行降噪处理，使用模型: {denoise_model}, 强度: {denoise_level}")
            # 创建降噪处理后的文件路径
            denoised_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"denoised_{int(time.time())}_{os.path.basename(file_path)}")
            
            # 执行降噪处理
            denoised_result = process_denoise(file_path, denoise_model, int(denoise_level))
            
            if denoised_result and os.path.exists(denoised_result):
                print(f"降噪成功，使用降噪后的文件进行识别: {denoised_result}")
                # 使用降噪后的文件
                file_path = denoised_result
            else:
                print(f"降噪失败，将使用原始文件进行识别")
        try:
            # 从文件中读取音频数据
            audio_bytes, _ = (
                ffmpeg.input(file_path, threads=0)
                .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            
            # 使用全局时间戳模型获取说话人分离信息
             # 使用全局时间戳模型获取说话人分离信息
            timestamp_res = timestamp_model.generate(
    input=audio_bytes, 
    batch_size_s=300, 
    is_final=True, 
    sentence_timestamp=True,
    speaker_change=True,     # 添加说话人变化检测
    max_speaker_num=10,# 设置最大说话人数量
    add_punc=True  # 确保此参数设置为True       
)
            timestamp_result = timestamp_res[0]
            # 添加调试输出
            # 打印原始结果以便调试
            print(f"时间戳模型返回结果: {timestamp_result}")
            print(f"结果包含的键: {list(timestamp_result.keys())}")

            print(f"原始说话人信息: {[s['spk'] for s in timestamp_result['sentence_info']]}")
            print(f"不同说话人数量: {len(set([s['spk'] for s in timestamp_result['sentence_info']]))}")
            # 提取时间戳、说话人和文本信息
            asr_result_text = timestamp_result['text']
            sentences = []
            
            for sentence in timestamp_result["sentence_info"]:
                start = to_date(sentence["start"])
                end = to_date(sentence["end"])
                if sentences and sentence["spk"] == sentences[-1]["spk"] and len(sentences[-1]["text"]) < int(split_number):
                    sentences[-1]["text"] += "" + sentence["text"]
                    sentences[-1]["end"] = end
                else:
                    sentences.append(
                        {"text": sentence["text"], "start": start, "end": end, "spk": sentence["spk"]}
                    )
            
            # 如果需要，使用选定的ASR模型替换文本内容（可选）
            if model_key != default_model and model_key != "sensevoice":
                try:
                    current_model = load_model(model_key)
                    # 用方言模型的整体识别结果替换
                    text_res = current_model.generate(input=audio_bytes, batch_size_s=300,batch_size=1)
                    print(f"text_res: {text_res}.keys()={text_res[0].keys()}")
                    asr_result_text = text_res[0]['text']  # 替换整体文本
                except Exception as e:
                    print(f"模型处理异常: {e}")
                    # 继续使用时间戳模型的文本结果
            
            # 创建处理结果目录
            date = datetime.now().strftime("%Y-%m-%d")
            output_path = os.path.join(output_folder, date, audio_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 剪切和处理每个句子段
            speaker_audios = {}  # 存储每个说话人的音频片段
            all_segments = []  # 存储所有分离的片段信息
            i = 0
            for stn in sentences:
                stn_txt = stn['text']
                start = stn['start']
                end = stn['end']
                spk = stn['spk']
                
                # 为每个说话人创建目录
                spk_path = os.path.join(output_path, str(spk))
                os.makedirs(spk_path, exist_ok=True)
                
                # 保存音频片段
                if file_ext in support_audio_format:
                    final_save_file = os.path.join(spk_path, f"{i}{file_ext}")
                    (
                        ffmpeg.input(file_path, threads=0, ss=start, to=end)
                        .output(final_save_file)
                        .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                             capture_stderr=True)
                    )
                elif file_ext in support_video_format:
                    final_save_file = os.path.join(spk_path, f"{i}.mp4")
                    (
                        ffmpeg.input(file_path, threads=0, ss=start, to=end)
                        .output(final_save_file, vcodec='libx264', crf=23, acodec='aac', ab='128k')
                        .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                             capture_stderr=True)
                    )
                
                # 添加情感识别部分 - 在上面代码后添加
                # 对分离的音频片段进行情感识别
                print(f"对片段 {i} 进行情感识别...")
                emotion_info = recognize_emotion(final_save_file)

                # 保存文本信息
                spk_txt_file = os.path.join(output_path, f'spk{spk}.txt')
                with open(spk_txt_file, 'a', encoding='utf-8') as f:
                    # 添加情感信息到文本
                    f.write(f"{start} --> {end}\n{stn_txt} {emotion_info['emotion']}\n\n")

                # 记录这个片段的信息
                segment_info = {
                    'id': i,
                    'spk': spk,
                    'start': start or "00:00:00.000",  # 确保有默认值
                    'end': end or "00:00:00.000",      # 确保有默认值
                    'text': clean_text(stn_txt) if stn_txt else "",  # 确保文本不为None
                    'file': final_save_file,
                    # 修改URL生成方式，添加错误处理
                    'url': f"/audio/{date}/{audio_name}/{spk}/{i}{file_ext if file_ext in support_audio_format else '.mp4'}",
                    # 添加情感分析结果
                    'emotion': emotion_info['emotion'],
                    'events': emotion_info['events']
                }
                all_segments.append(segment_info)
                
                # 记录说话人和对应的音频片段
                if spk not in speaker_audios:
                    speaker_audios[spk] = []
                speaker_audios[spk].append(segment_info)
                i += 1
            
            # 注释掉或删除合并代码
            # for spk, audio_segments in speaker_audios.items():
            #    output_file = os.path.join(output_path, f"{spk}.mp3")
            #    inputs = [seg['file'] for seg in audio_segments]
            #    concat_audio = AudioSegment.from_file(inputs[0])
            #    for i in range(1, len(inputs)):
            #        concat_audio = concat_audio + AudioSegment.from_file(inputs[i])
            #    concat_audio.export(output_file, format="mp3")
            
            # 重新识别每个句子并更新文本
            for i, stn in enumerate(sentences):
                # 从原始音频中提取这个句子的音频片段
                segment_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{i}.wav")
                # 提取音频片段
                (
                    ffmpeg.input(file_path, threads=0, ss=stn['start'], to=stn['end'])
                    .output(segment_file, acodec='pcm_s16le', ac=1, ar=16000)
                    .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                        capture_stderr=True)
                )
                
                # 使用选定模型识别并更新文本
                try:
                    current_model = load_model(model_key)
                    stn_res = current_model.generate(input=segment_file, batch_size_s=300,batch_size=1)
                    cleaned_text = clean_text(stn_res[0]['text'])
                    sentences[i]['text'] = cleaned_text
                    
                    # 更新分段文本
                    for segment in all_segments:
                        if segment['id'] == i and segment['spk'] == stn['spk']:
                            segment['text'] = cleaned_text
                            break
                    
                    print(f"使用 {models[model_key]['name']} 模型成功识别分段 {i}, 说话人 {stn['spk']}")
                except Exception as e:
                    print(f"模型处理异常: {e}")
            
            # 关键修改: 重新合成总体文本
            # 在所有分段都重新识别后，根据分段顺序重组总体文本
            combined_text = ""
            # 按时间顺序排序所有片段
            sorted_segments = sorted(all_segments, key=lambda x: x['start'] if x['start'] else "")
            
            # 合并所有片段的文本
            for segment in sorted_segments:
                if segment['text']:
                    combined_text += segment['text'] + " "
            
            # 去除多余空格并清理
            asr_result_text = clean_text(combined_text)
            
            # 返回更新后的结果
            result = {
                "status": "success", 
                "message": f"使用 {models[model_key]['name']} 模型处理完成",
                "text": asr_result_text,  # 注意这里使用了更新后的文本
                "segments": all_segments or [],
                "speakers": list(speaker_audios.keys()) if speaker_audios else [],
                "model": models[model_key]['name'],
                "duration": to_milliseconds(sentences[-1]['end']) if sentences else 0
            }
            
            # 如果使用了降噪，添加到返回结果中
            if enable_denoise:
                result["denoised"] = True
                result["denoise_model"] = denoise_model
                
            return result
            
        except Exception as e:
            # 捕获所有异常，记录并返回错误信息
            error_message = str(e)
            print(f"音频处理异常: {error_message}")
            return {"status": "error", "message": f"处理失败: {error_message}"}
        
        finally:
            # 清理临时文件
            for i in range(len(sentences) if 'sentences' in locals() else 0):
                temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{i}.wav")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"处理失败: {str(e)}",
            "text": "",
            "segments": [],
            "speakers": [],
            "model": models[model_key]["name"],
            "duration": 0
        })

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/audio/<path:filepath>')
def get_audio(filepath):
    """提供对音频文件的访问"""
    try:
        # 打印路径信息以便调试
        print(f"访问音频: {filepath}")
        print(f"完整路径: {os.path.join(app.config['OUTPUT_FOLDER'], filepath)}")
        
        # 检查文件是否存在
        full_path = os.path.join(app.config['OUTPUT_FOLDER'], filepath)
        if not os.path.exists(full_path):
            print(f"警告: 文件不存在 {full_path}")
            return jsonify({"error": "文件不存在"}), 404
            
        return send_from_directory(app.config['OUTPUT_FOLDER'], filepath)
    except Exception as e:
        print(f"音频访问异常: {e}")
        return jsonify({"error": str(e)}), 500
# 添加降噪页面路由
@app.route('/denoise_page')
def denoise_page():
    """返回降噪工具页面"""
    return render_template('denoise.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "没有找到文件"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"})
    
    # 保存上传文件
    filename = werkzeug.utils.secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 获取分离字数
    try:
        split_number = int(request.form.get('split_number', 10))
    except ValueError:
        split_number = 10  # 默认值
    
    # 获取选定的模型
    model_key = request.form.get('model', default_model)
    if model_key not in models:
        model_key = default_model

    # 获取降噪参数
    enable_denoise = request.form.get('enable_denoise') == 'true'
    denoise_model = request.form.get('denoise_model', 'HP5')
    try:
        denoise_level = int(request.form.get('denoise_level', 10))
    except ValueError:
        denoise_level = 10
    
    try:
        # 处理文件，添加降噪参数
        result = process_audio(
            file_path, 
            app.config['OUTPUT_FOLDER'], 
            model_key, 
            split_number,
            enable_denoise,
            denoise_model,
            denoise_level
        )
        
        # 确保返回值中的所有字段都有有效值
        if "status" not in result:
            result["status"] = "error"
        if "message" not in result:
            result["message"] = "未知错误"
        if "text" not in result or result["text"] is None:
            result["text"] = ""
        if "segments" not in result or result["segments"] is None:
            result["segments"] = []
        if "speakers" not in result or result["speakers"] is None:
            result["speakers"] = []
        if "model" not in result:
            result["model"] = models[model_key]["name"]
        if "duration" not in result:
            result["duration"] = 0
        
        return jsonify(result)
    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"处理失败: {str(e)}",
            "text": "",
            "segments": [],
            "speakers": [],
            "model": models[model_key]["name"],
            "duration": 0
        })
@app.route('/clone', methods=['POST'])
def clone_voice():
    """使用CosyVoice克隆说话人声音"""
    if not cosyvoice_available or cosyvoice_model is None:
        return jsonify({"status": "error", "message": "声音克隆功能不可用，CosyVoice模型未加载"})
    
    # 获取请求参数
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "请求格式错误，需要JSON数据"})
    
    # 必需参数：说话人ID、要合成的文本、日期、音频名称
    speaker_id = data.get('speaker_id')
    text = data.get('text')
    date = data.get('date')
    audio_name = data.get('audio_name')
    
    if not all([speaker_id, text, date, audio_name]):
        return jsonify({"status": "error", "message": "缺少必要参数: speaker_id, text, date, audio_name"})
    
    try:
        # 构建该说话人音频文件夹路径
        speaker_folder = os.path.join(app.config['OUTPUT_FOLDER'], date, audio_name, str(speaker_id))
        if not os.path.exists(speaker_folder):
            return jsonify({"status": "error", "message": f"找不到说话人音频文件夹: {speaker_folder}"})
        
        # 获取第一个音频文件作为参考声音
        audio_files = [f for f in os.listdir(speaker_folder) if f.endswith(tuple(support_audio_format))]
        if not audio_files:
            audio_files = [f for f in os.listdir(speaker_folder) if f.endswith('.mp4')]
        
        if not audio_files:
            return jsonify({"status": "error", "message": f"说话人文件夹中没有音频文件: {speaker_folder}"})
        
        # 使用第一个音频作为参考
        prompt_path = os.path.join(speaker_folder, audio_files[0])
        print(f"使用音频文件作为声音参考: {prompt_path}")
        
        # 从文件名获取片段ID
        segment_id = os.path.splitext(os.path.basename(prompt_path))[0]
        print(f"参考音频片段ID: {segment_id}")
        
                # 在函数clone_voice中修改参考文本获取逻辑
        spk_txt_file = os.path.join(app.config['OUTPUT_FOLDER'], date, audio_name, f'spk{speaker_id}.txt')
        reference_text = "这是一段参考音频"  # 默认文本，仅在识别失败时使用

        # 直接使用语音识别模型识别参考音频
        print(f"开始直接识别参考音频: {prompt_path}")
        try:
            # 处理音频格式，确保为16k单声道
            temp_wav = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_rec_{uuid.uuid4()}.wav")
            (
                ffmpeg.input(prompt_path, threads=0)
                .output(temp_wav, format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            
            # 读取处理后的音频
            with open(temp_wav, 'rb') as f:
                audio_bytes = f.read()
            
            # 使用默认模型进行识别
            current_model = load_model(default_model)
            rec_result = current_model.generate(
                input=audio_bytes, 
                batch_size=1,
                batch_size_s=60
            )
            
            # 从识别结果中提取文本
            if rec_result and len(rec_result) > 0 and 'text' in rec_result[0]:
                recognized_text = rec_result[0]['text']
                reference_text = clean_text(recognized_text)
                print(f"参考音频识别成功，文本: {reference_text}")
            else:
                print(f"参考音频识别返回无效结果，使用默认文本")
            
            # 清理临时文件
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        except Exception as e:
            print(f"参考音频识别失败: {e}")
            print("将使用默认参考文本")
        # 如果是mp4文件，先提取音频
        if prompt_path.endswith('.mp4'):
            temp_wav = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_prompt_{uuid.uuid4()}.wav")
            (
                ffmpeg.input(prompt_path)
                .output(temp_wav, format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            prompt_path = temp_wav
        
        
                
        # 加载prompt音频
        prompt_speech_16k = load_wav(prompt_path, 16000)
        
        # 创建保存目录
        clone_folder = os.path.join(app.config['OUTPUT_FOLDER'], date, audio_name, 'cloned')
        os.makedirs(clone_folder, exist_ok=True)
        
        # 生成克隆语音
        output_filename = f"cloned_{speaker_id}_{int(time.time())}.wav"
        output_path = os.path.join(clone_folder, output_filename)
        
        # 在找到参考文本后，清理所有表情符号
        reference_text = clean_emoji_text(reference_text)
        print(f"清理表情后的参考文本: {reference_text}")

        # 执行声音克隆时使用清理后的文本
        print(f"开始声音克隆，参考文本: {reference_text}, 合成文本: {text}")
        clone_generator = cosyvoice_model.inference_zero_shot(
            text,            # tts_text: 要合成的新文本
            reference_text,  # 已清理表情的参考文本
            prompt_speech_16k,
            stream=False
        )
        
        # 方法1: 直接遍历生成器获取第一个结果
        for result in clone_generator:
            # 只处理第一个结果
            torchaudio.save(
                output_path, 
                result['tts_speech'], 
                cosyvoice_model.sample_rate
            )
            
            # 返回结果
            return jsonify({
                "status": "success",
                "message": "声音克隆成功",
                "audio_url": f"/audio/{date}/{audio_name}/cloned/{output_filename}",
                "speaker_id": speaker_id,
                "text": text
            })
        
        # 如果没有结果，会执行到这里
        return jsonify({
            "status": "error",
            "message": "声音克隆生成失败，没有结果返回"
        })
            
    except Exception as e:
        print(f"声音克隆异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"声音克隆失败: {str(e)}"
        })
    finally:
        # 清理临时文件
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass
@app.route('/outputs/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# 添加模型查询API

@app.route('/models', methods=['GET'])
def get_models():
    """返回所有可用模型的列表"""
    model_list = []
    for key, info in models.items():
        if key != "timestamp":  # 不显示时间戳模型
            model_list.append({
                "id": key,
                "name": info["name"],
                "default": key == default_model
            })
    return jsonify({"models": model_list})

# 在main中调用
if __name__ == '__main__':
    print("SenseVoice 说话人分离应用")
    initialize_app()  # 只在实际运行时才初始化
    app.run(debug=False, port=5000)