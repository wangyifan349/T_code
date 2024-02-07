"""
pip install tensorflow tensorflow_hub scipy
使用的预训练模型
YAMNet 是一个深度网络，可以从训练它的 AudioSet-YouTube 语料库中预测 521 个音频事件类。它采用 Mobilenet_v1 深度可分离卷积架构。
加载 YAMNet 模型，并读取指定的 WAV 文件
它将音频数据归一化到 [-1.0, 1.0] 的范围内，并使用模型对音频进行分类
最后可视化
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

# 可视化
plt.figure(figsize=(10, 6))

# 绘制波形图
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])
plt.title('波形图')

# 绘制对数梅尔频谱图
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
plt.title('对数梅尔频谱图')

# 绘制模型输出得分
mean_scores = np.mean(scores, axis=0)
top_n = 10
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
plt.ylim(-0.5 + np.array([top_n, 0]))
plt.title('模型输出得分')

plt.tight_layout()
plt.show()
