import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.

import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 确保使用的是TensorFlow 2.x 版本
print("TensorFlow version:", tf.__version__)

# 加载IMDb评论数据集
train_data, validation_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]'),
    as_supervised=True)

# 加载TensorFlow Hub中的BERT模型和预处理器
bert_preprocess_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)

# 构建模型
def build_classifier_model():
    # 输入文本
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    # 预处理文本
    preprocessed_text = bert_preprocess_model(text_input)
    # BERT编码器
    outputs = bert_model(preprocessed_text)
    # 使用BERT的汇集输出
    net = outputs['pooled_output']
    # Dropout层减少过拟合
    net = layers.Dropout(0.1)(net)
    # 输出层，二进制分类
    net = layers.Dense(1, activation=None, name='classifier')(net)
    # 构建模型
    return Model(text_input, net)
classifier_model = build_classifier_model()

# 编译模型
classifier_model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

# 训练配置
epochs = 3
batch_size = 32

# 数据集准备
def prepare_data(train_data, validation_data, batch_size):
    # 打乱训练数据以增加随机性，参数10000指定了打乱时使用的缓冲区大小。
    # 在实际应用中，这个数值可以根据你的数据集大小进行调整。
    train_data = train_data.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # 批量化验证数据。由于我们不需要在验证数据上进行打乱操作，
    # 所以这里没有调用shuffle。
    validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # prefetch(tf.data.AUTOTUNE)允许数据加载过程在模型训练时并行进行。
    # AUTOTUNE选项会让TensorFlow动态决定预加载的批次数量，
    # 这有助于减少CPU和GPU之间的空闲时间。
    return train_data, validation_data

# 调用prepare_data函数来准备好训练和验证数据集
train_data, validation_data = prepare_data(train_data, validation_data, batch_size)

# 训练模型
history = classifier_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs
)

# 评估模型
loss, accuracy = classifier_model.evaluate(validation_data)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
"""
TensorFlow Hub上的BERT预处理模型和BERT模型本身。
模型被设置为可训练的，这意味着BERT层的权重将在训练过程中更新。
使用了一个简单的全连接层来作为分类器的输出层，由于这是一个二分类任务，
所以没有使用激活函数并且损失函数是BinaryCrossentropy。
"""
