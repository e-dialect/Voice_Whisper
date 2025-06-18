import os
import sys
from pathlib import Path
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice')
sys.path.append('/home/chihan/workspace/SenseVoice/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 使用CosyVoice2（推荐）
cosyvoice = CosyVoice2('/home/chihan/workspace/SenseVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('这是一段文本', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)