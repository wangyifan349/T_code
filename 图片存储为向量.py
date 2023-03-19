import os
import tensorflow as tf
import numpy as np

def image_to_vector(image_path, model):
    """
    将指定路径的图像转换成向量
    """
    # 加载图像并将其调整为模型所需的输入大小
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    # 将图像转换为Numpy数组，并按照ResNet50模型的要求进行预处理
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.resnet.preprocess_input(x)
    
    # 使用给定的模型将处理后的图像转换为特征向量
    features = model.predict(tf.expand_dims(x, axis=0))[0]
    
    return features

# 指定图像所在的文件夹和模型
image_dir = "path/to/image/folder"
model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

# 获取文件夹中所有图片的文件名列表
image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

# 使用ResNet50模型将每个图像转换成向量，并将向量保存到一个Numpy数组中
vectors = []
for name in image_names:
    path = os.path.join(image_dir, name)
    
    # 将每个图像转换为特征向量，并将所有向量保存到一个列表中
    vector = image_to_vector(path, model)
    vectors.append(vector)
vectors = np.array(vectors)

# 将保存有向量的Numpy数组保存到磁盘上
np.save("image_vectors.npy", vectors)
