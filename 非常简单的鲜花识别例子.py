import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
#############################
# 获取所有可用的物理GPU设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存动态增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 获取逻辑GPU设备信息
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # 打印GPU信息
        print(f"发现 {len(gpus)} 块物理GPU, {len(logical_gpus)} 块逻辑GPU")
        # 选择最大的GPU
        largest_gpu = None
        largest_memory = 0
        for gpu in gpus:
            gpu_memory = tf.config.experimental.get_memory_info(gpu).size
            if gpu_memory > largest_memory:
                largest_gpu = gpu
                largest_memory = gpu_memory
        if largest_gpu is not None:
            tf.config.experimental.set_virtual_device_configuration(
                largest_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)]
            )
            print(f"选择了最大的GPU，其内存大小为 {largest_memory / (1024 ** 3)} GB")
    except RuntimeError as e:
        print(e)
else:
    print("未发现可用的GPU设备")
##########################################
# Define the path to your custom dataset directory
dataset_directory = 'path_to_your_custom_dataset/'
"""
#假设目录是这样的
- dataset_directory/
  - class1/
    - image1.jpg
    - image2.jpg
    - ...
  - class2/
    - image1.jpg
    - image2.jpg
    - ...
  - ...
"""
# 导入图像数据生成器
train_datagen = ImageDataGenerator(
    rotation_range=20,  # 随机旋转图像的角度范围，单位为度
    width_shift_range=0.2,  # 随机水平平移图像的比例
    height_shift_range=0.2,  # 随机垂直平移图像的比例
    shear_range=0.2,  # 随机剪切图像的强度
    zoom_range=0.2,  # 随机缩放图像的范围
    horizontal_flip=True,  # 随机水平翻转图像
    fill_mode='nearest',  # 用于填充新像素的策略，'nearest' 表示最近邻插值
    rescale=1.0/255.0  # 对图像像素值进行重新缩放，将其归一化到 0 到 1 之间
)


# Specify the target image size
target_size = (224, 224)

# Create a data generator for training data from your custom dataset
train_generator = train_datagen.flow_from_directory(
    dataset_directory,  # 数据集目录，存储图像数据的文件夹路径
    target_size=target_size,  # 目标图像尺寸，通常是一个元组，例如(224, 224)，用于将所有图像调整为相同的尺寸
    batch_size=32,  # 批量大小，即每个训练步骤从数据生成器中产生的图像数量，可以根据需要调整此值
    class_mode='categorical',  # 类别模式，如果您的数据集有多个类别，应该选择'categorical'，如果只有两个类别可以选择'binary'
    shuffle=True  # 是否对数据进行随机洗牌，洗牌数据可以增加训练的随机性
)

"""
对于图像分类任务，卷积神经网络通常是一个不错的选择，因为它们具有天然的特征提取能力。
常见的 CNN 架构如VGG、ResNet和Inception等可以作为参考模型，可以根据您的任务进行修改和微调。
"""
# Build and compile your CNN model (similar to previous code)
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Replace num_classes with the number of flower classes in your dataset
])
# 创建一个 ModelCheckpoint 回调
model_checkpoint = ModelCheckpoint(
    'custom_flower_recognition_model.h5',  # 指定保存模型的文件名
    save_best_only=True,  # 只保存性能最佳的模型
    monitor='val_accuracy',  # 监视验证集准确度
    mode='max',  # 监视模式，这里是最大化验证集准确度
    save_freq='epoch',  # 每个 epoch 保存一次模型
    period=50  # 每训练 50 个 epoch 就保存一次
)
# 通用训练保存
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for multiple classes
              metrics=['accuracy'])
#history = model.fit(train_generator, epochs=10)
# 训练模型时将回调传递给 fit 方法
history = model.fit(
    train_generator,
    epochs=epochs,
    callbacks=[model_checkpoint],  # 添加 ModelCheckpoint 回调
)
model.save('custom_flower_recognition_model.h5')
# 重新加载这个模型
model = load_model('flower_recognition_model.h5')
#............其他部分代码
