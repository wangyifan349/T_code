import tensorflow as tf
from tensorflow.keras import layers
import os
# 创建生成模型
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    # 第一层是全连接层，没有偏置，输入是潜在空间的维度
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
    # 批量归一化层，用于稳定训练过程
    model.add(layers.BatchNormalization())
    # LeakyReLU激活函数，避免梯度消失问题
    model.add(layers.LeakyReLU())
    # 重塑层，将一维数据转换为三维数据，为卷积层做准备
    model.add(layers.Reshape((7, 7, 256)))
    # 反卷积层，用于上采样图像
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # 继续上采样，增加图像尺寸
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # 最后一层反卷积，生成最终图像，使用tanh激活函数，输出范围为[-1, 1]
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# 创建判别模型
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    # 卷积层，用于提取图像特征
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    # Dropout层，减少过拟合
    model.add(layers.Dropout(0.3))
    # 第二个卷积层，进一步提取特征
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # 展平层，将三维特征图展平为一维
    model.add(layers.Flatten())
    # 输出层，一个神经元，用于判断输入图像是真是假
    model.add(layers.Dense(1))
    return model

# 定义GAN模型
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_images):
        # 随机噪声样本
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 生成假图像
        generated_images = self.generator(random_latent_vectors)

        # 组合真假图像
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # 组合真假标签
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # 添加随机噪声到标签 - 标签平滑
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 训练判别器
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 训练生成器 (我们希望判别器误将这些图像看作是真的)
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

# 训练GAN
def train_gan(gan, dataset, epochs):
    for epoch in range(epochs):
        for img_batch in dataset:
            gan.train_step(img_batch)

latent_dim = 128
generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28, 1))

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

# 编译GAN
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)


# 设置代理服务器
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
# 加载和预处理MNIST数据集
(img_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
#(img_train, _), (_, _) = tf.keras.datasets.mnist.load_data(path='C:/Users/WWW/Downloads/mnist.npz')

#(img_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img_train = img_train[:2000]#假设不用全部数据
img_train = img_train.reshape(img_train.shape[0], 28, 28, 1).astype("float32")
img_train = (img_train - 127.5) / 127.5 # 将图像缩放到[-1, 1]

# 批量和打乱数据
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(img_train).shuffle(1000).batch(batch_size)
# 训练模型
epochs = 60 # 设置一个典型的样本轮数，实际可能需要更多轮次
train_gan(gan, train_dataset, epochs)
# 保存生成器模型
generator.save('./generator_model.h5')

# 使用生成器生成图片
generated_images = generator(random_latent_vectors)

# 将生成的图片转换回合适的像素值范围 [0, 255] 并转为整数
generated_images = (generated_images * 127.5) + 127.5
generated_images = generated_images.numpy().astype('uint8')

# 显示生成的图片
#import matplotlib.pyplot as plt
import numpy as np
#for i in range(10):
    #plt.subplot(2, 5, i + 1)
    #plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    #plt.axis('off')
#plt.show()
for i in range(10):
    # 将图像转换为PIL格式
    img = Image.fromarray(generated_images[i].reshape(28, 28))
    # 保存图像
    img.save(f'generated_image_{i}.png')
