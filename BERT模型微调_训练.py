import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# TensorFlow Hub BERT模型的URL
# BERT预处理和编码器的URL
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# 数据集加载
squad_builder = tfds.builder("squad")
squad_builder.download_and_prepare()

# 加载数据集
train_data = squad_builder.as_dataset(split="train[:87%]")
validation_data = squad_builder.as_dataset(split="train[87%:]")

# BERT特征提取器
bert_preprocessor = hub.KerasLayer(tfhub_handle_preprocess)

# 构建并编译模型
def create_qa_model():
    """
    创建用于问答任务的模型
    """
    # 输入层：接收字符串
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    
    # 预处理层
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    
    # 编码器层：BERT模型
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    
    # 使用序列输出来预测答案位置
    sequence_output = outputs['sequence_output']
    
    # 为答案的起始和结束位置添加输出层
    start_logits = tf.keras.layers.Dense(1, name='start_logit')(sequence_output)
    start_logits = tf.keras.layers.Flatten()(start_logits)  # 展平为一维数组
    end_logits = tf.keras.layers.Dense(1, name='end_logit')(sequence_output)
    end_logits = tf.keras.layers.Flatten()(end_logits)      # 展平为一维数组
    
    # 使用softmax层来将logits转换为概率
    start_probs = tf.keras.layers.Activation(tf.nn.softmax, name='start_probs')(start_logits)
    end_probs = tf.keras.layers.Activation(tf.nn.softmax, name='end_probs')(end_logits)
    
    # 创建模型
    model = tf.keras.Model(inputs=text_input, outputs=[start_probs, end_probs])
    
    return model

def input_fn(data):
    """
    数据输入函数
    """
    # 提取问题、上下文和答案的起始和结束位置
    def extract_fn(data):
        return {
            "text": tf.cast(data["context"], tf.string),
        }, {
            "start_probs": tf.cast(data["start_positions"], tf.float32),
            "end_probs": tf.cast(data["end_positions"], tf.float32)
        }

    # 将数据集映射到输入和输出
    dataset = data.map(extract_fn)
    return dataset

# 创建模型
qa_model = create_qa_model()

# 打印模型摘要
qa_model.summary()

# 输入输出示例
text = tf.constant(["The quick brown fox jumps over the lazy dog.",
                    "The dog slept on the verandah."])

start_probs, end_probs = qa_model.predict(text)

print("Start probabilities shape:", start_probs.shape)
print("End probabilities shape:", end_probs.shape)

# 显示模型结构和输入输出示例
print(qa_model)

# 训练模型
def train_model(model, train_dataset, validation_dataset, epochs=2, batch_size=32, steps_per_epoch=1000):
    """
    训练模型
    """
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # 打印关键参数
    print("Training parameters:")
    print("====================")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print("====================")
    
    # 训练模型
    print("Training started...")
    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=10, # 这里添加了validation_steps参数
                        verbose=1) # 添加了verbose参数以显示训练进度信息
    print("Training completed.")
    return history

# 转换数据集
train_dataset = input_fn(train_data).batch(32)
validation_dataset = input_fn(validation_data).batch(32)

# 训练模型
history = train_model(qa_model, train_dataset, validation_dataset, epochs=2, batch_size=32, steps_per_epoch=1000)
