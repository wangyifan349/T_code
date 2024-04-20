import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载YamNet模型
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
yamnet_classes = np.array([
    line.rstrip() for line in tf.keras.utils.get_file('yamnet_class_map.csv',
                                                     'https://storage.googleapis.com/ml-research/paper-implementation/yamnet/yamnet_class_map.csv')
])
# 将音频文件加载为Mel频谱图
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(audio, sr=sr)
    # Pad audio to one second for consistent input length
    if duration < 1.0:
        audio = np.pad(audio, (0, int(np.ceil((1.0 - duration) * sr))), mode='constant')
    elif duration > 1.0:
        audio = audio[:sr]
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel_spec

# 音频文件路径
audio_file_path = 'your_audio_file.wav'

# 加载音频文件并转换为Mel频谱图
mel_spec = load_audio(audio_file_path)

# 将频谱图调整为模型的输入大小
input_batch = tf.convert_to_tensor(mel_spec)
input_batch = tf.expand_dims(input_batch, 0)
input_batch = tf.expand_dims(input_batch, -1)

# 使用模型进行预测
scores, embeddings, spectrogram = yamnet_model(input_batch)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)

# 获取预测结果
predicted_class = yamnet_classes[top_class.numpy()]

# 打印预测结果
print("Predicted class:", predicted_class)

# 可视化Mel频谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
