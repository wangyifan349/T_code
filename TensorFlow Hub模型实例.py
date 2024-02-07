# 导入所需的库
pip install --quiet "tensorflow-text==2.8.*"

import seaborn as sns  # 数据可视化库
from sklearn.metrics import pairwise  # 用于计算配对样本间的度量

import tensorflow as tf  # TensorFlow 深度学习库
import tensorflow_hub as hub  # TensorFlow Hub
import tensorflow_text as text  # 用于文本处理的 TensorFlow 库

# 配置模型
sentences = [
  "Here We Go Then, You And I is a 1999 album by Norwegian pop artist Morten Abel. It was Abel's second CD as a solo artist.",
  "The album went straight to number one on the Norwegian album chart, and sold to double platinum.",
  "Among the singles released from the album were the songs \"Be My Lover\" and \"Hard To Stay Awake\".",
  "Riccardo Zegna is an Italian jazz musician.",
  "Rajko Maksimović is a composer, writer, and music pedagogue.",
  "One of the most significant Serbian composers of our time, Maksimović has been and remains active in creating works for different ensembles.",
  "Ceylon spinach is a common name for several plants and may refer to: Basella alba Talinum fruticosum",
  "A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby totally or partly obscuring the image of the Sun for a viewer on Earth.",
  "A partial solar eclipse occurs in the polar regions of the Earth when the center of the Moon's shadow misses the Earth.",
]

# 运行模型
# 从 TF-Hub 加载 BERT 模型，使用 TF-Hub 中的匹配预处理模型将句子词例化，然后将词例化句子馈入模型。为了让此 Colab 变得快速而简单，我们建议在 GPU 上运行。
hub模型 = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
预处理模型 = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

preprocess = hub.load(预处理模型)
bert = hub.load(hub模型)
inputs = preprocess(sentences)
outputs = bert(inputs)

print("句子:")
print(sentences)

print("\nBERT 模型输入:")
print(inputs)

print("\n汇总嵌入:")
print(outputs["汇总输出"])

print("\n每个词的嵌入:")
print(outputs["序列输出"])
