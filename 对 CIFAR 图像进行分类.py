# 导入所需的库
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 加载 CIFAR-10 数据集，并将像素值归一化到 0 到 1 之间
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 展示部分训练集中的图像及其标签
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # CIFAR-10 数据集中的标签是数组，因此需要额外的索引
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# 构建卷积神经网络模型
model = models.Sequential()
# 添加第一层卷积层，32个3x3的卷积核，relu激活函数，输入图像大小为32x32x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加第二层卷积层，64个3x3的卷积核，relu激活函数
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 再次添加最大池化层
model.add(layers.MaxPooling2D((2, 2)))
# 添加第三层卷积层，64个3x3的卷积核，relu激活函数
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 在模型上添加密集连接分类器
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 保存模型权重
model.save_weights('cifar10_cnn_model_weights.h5')

# 使用模型进行预测
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 随机选择一张测试图像进行预测
index = np.random.randint(0, len(test_images))
predicted_label = np.argmax(predictions[index])
true_label = test_labels[index][0]
predicted_class = class_names[predicted_label]
true_class = class_names[true_label]

# 打印预测结果
print("Predicted class:", predicted_class)
print("True class:", true_class)

# 显示预测的图像
plt.figure()
plt.imshow(test_images[index])
plt.colorbar()
plt.grid(False)
plt.show()
