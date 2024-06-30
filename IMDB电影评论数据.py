import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 加载IMDB电影评论数据集
(train_data, test_data), info = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True, with_info=True)

# 定义批量大小
batch_size = 32

# 将数据集分批次
train_data = train_data.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 使用TensorFlow Hub中的预训练文本嵌入模块
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"

# 构建模型
model = tf.keras.Sequential([
    hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_data,
                    epochs=5,
                    validation_data=test_data)

# 评估模型
results = model.evaluate(test_data)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# 保存模型权重
model.save_weights('sentiment_model_weights')

# 加载模型权重
model.load_weights('sentiment_model_weights')

# 使用加载的模型进行预测
sample_text = ["This movie was fantastic! I really enjoyed it."]
predictions = model.predict(sample_text)
print(predictions)
