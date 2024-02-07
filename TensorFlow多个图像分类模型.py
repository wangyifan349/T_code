#使用 TensorFlow 进行推断。您选择了 EfficientNetV2 模型，这是一个很好的选择，因为它已经在 ImageNet 数据集上训练过了

import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

# 选择一个图像分类模型
model_handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"

# 加载模型
classifier = hub.load(model_handle)

# 加载图像
image = load_image("your_image.jpg")  # 请替换为您自己的图像文件名

# 运行模型推断
probabilities = tf.nn.softmax(classifier(image)).numpy()

# 获取排名前 5 的结果
top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
np_classes = np.array(classes)

includes_background_class = probabilities.shape[1] == 1001

for i, item in enumerate(top_5):
    class_index = item if includes_background_class else item + 1
    line = f'({i+1}) {class_index:4} - {classes[class_index]}: {probabilities[0][top_5][i]}'
    print(line)

# 显示图像
show_image(image, '')
