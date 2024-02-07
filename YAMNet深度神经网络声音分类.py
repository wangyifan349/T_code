"""
pip install tensorflow tensorflow_hub scipy
使用的预训练模型
YAMNet 是一个深度网络，可以从训练它的 AudioSet-YouTube 语料库中预测 521 个音频事件类。它采用 Mobilenet_v1 深度可分离卷积架构。
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile
from IPython.display import Audio

# 确保 TensorFlow 的版本至少是 2
assert tf.__version__.startswith('2')

# 加载 YAMNet 模型
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
model = hub.load(yamnet_model_handle)

# 定义一个函数来获取 YAMNet 的类名
def class_names_from_csv(class_map_csv_text):
    """从 CSV 文件中读取类名."""
    class_names = []
    with open(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

# 获取 YAMNet 的类名
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = class_names_from_csv(class_map_path)

# 选择一个 WAV 文件进行测试
wav_file_name = 'miaow_16k.wav'  # 或者 'speech_whistling2.wav'

# 读取 WAV 文件
sample_rate, wav_data = wavfile.read(wav_file_name)

# 显示一些基本信息
duration = len(wav_data) / sample_rate
print(f'采样率: {sample_rate} Hz')
print(f'总时长: {duration:.2f}s')
print(f'输入大小: {len(wav_data)}')

# 归一化 wav_data 到 [-1.0, 1.0]
waveform = wav_data / np.iinfo(np.int16).max

# 运行模型，获取输出
scores, embeddings, spectrogram = model(waveform)

# 处理模型输出
scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
inferred_class = class_names[scores_np.mean(axis=0).argmax()]

# 打印主要声音类别
print(f'主要声音是: {inferred_class}')
