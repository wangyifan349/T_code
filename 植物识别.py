import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

# 数据预处理部分
# 设置图片大小
IMG_SIZE = (224, 224)
# 设置批次大小
BATCH_SIZE = 32
# 设置文件夹路径，此处请填写您的文件夹路径
data_dir = 'path/to/your/plant/folder'

# 使用ImageDataGenerator对图片进行预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 缩放图片
    rotation_range=40,  # 随机旋转
    width_shift_range=0.2,  # 随机水平位移
    height_shift_range=0.2,  # 随机垂直位移
    shear_range=0.2,  # 随机剪切
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True,  # 随机水平翻转
    fill_mode='nearest',  # 图片填充
    validation_split=0.2  # 划分验证集
)
#ImageDataGenerator 是一个非常有用的 TensorFlow Keras 工具，用于图像数据的预处理和实时数据增强。这个类通过实时数据增强生成的批次数据来训练模型，可以大大提高模型的泛化能力。





#rescale: 一个缩放因子，用于对图像进行缩放。通常，我们将此值设置为 1/255，将像素值缩放到 0-1 之间，这有助于神经网络学习。

#rotation_range: 整数值，用于随机旋转图像的度数范围。例如，如果设置为 40，则表示图像将在 -40 到 40 度之间随机旋转。

#width_shift_range 和 height_shift_range: 浮点数（小于 1 的值表示比例，大于 1 的值表示像素值），用于随机平移图像的水平和垂直范围。例如，如果设置为 0.2，表示图像在宽度/高度的 20% 范围内随机平移。

#shear_range: 浮点数，用于随机应用剪切变换（错切变换）。剪切角度在弧度制表示。例如，如果设置为 0.2，则表示图像将应用最多 0.2 弧度的剪切变换。

#zoom_range: 浮点数或 [lower, upper] 格式的浮点数列表，用于随机缩放图像。如果为浮点数，则表示缩放范围为 [1-zoom_range, 1+zoom_range]。例如，如果设置为 0.2，则表示图像在 80% 到 120% 之间随机缩放。

#horizontal_flip: 布尔值，表示是否随机水平翻转图像。

#fill_mode: 字符串，用于指定填充新创建像素的方法。可选值有 "constant"、"nearest"、"reflect" 和 "wrap"。默认为 "nearest"。

#validation_split: 浮点数，用于在训练集和验证集之间划分数据。例如，如果设置为 0.2，则表示将 20% 的数据用作验证集，其余 80% 的数据用作训练集。

#使用 ImageDataGenerator 后，您还需要使用 flow_from_directory 方法从指定的目录生成数据。这个方法的一些关键参数如下：

#directory: 数据集所在的目录路径。

#target_size: 一个整数元组 (height, width)，用于调整所有图像的尺寸。

#batch_size: 用于生成数据批次的大小。

#class_mode: 字符串，表示返回的标签数组的类型。可选值有 "categorical"、"binary"、"


#

"sparse" 和 "input"。这里是对这些值的解释：

#categorical: 对于多分类问题，返回 one-hot 编码后的标签。例如，对于 3 个类别，对应的 one-hot 编码标签可能为 [1, 0, 0]、[0, 1, 0] 和 [0, 0, 1]。

#binary: 用于二分类问题，返回单个二进制值的标签。例如，对于 2 个类别，对应的标签可能为 0 和 1。

#sparse: 返回整数格式的标签。例如，对于 3 个类别，对应的标签可能为 0、1 和 2。这种方式相比 categorical 更节省内存，但需要在模型定义时使用 SparseCategoricalCrossentropy 损失函数。

#input: 返回与输入图像相同的标签。这在自编码器等无监督学习场景中有用。

#None: 不返回任何标签，仅返回输入图像。这在预测或仅需提取图像特征的情况下有用。



# 生成训练集和验证集
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 创建神经网络模型部分
# 构建卷积神经网络
model = tf.keras.models.Sequential([
    # 添加卷积层
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # 添加全连接层
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# 保存模型
model.save('plant_recognition_model.h5')
