#!/bin/bash
# filepath: extract_features.sh

# 提取说话人嵌入
tools/extract_embedding.py --dir data/northeastern \
  --onnx_path pretrained_models/CosyVoice2-0.5B/campplus.onnx

# 提取语音token
tools/extract_speech_token.py --dir data/northeastern \
  --onnx_path pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx