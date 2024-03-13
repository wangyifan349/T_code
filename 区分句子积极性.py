import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 准备数据
positive_sentences = ['我很开心', '今天天气很好', '这是一个美好的世界']
normal_sentences = ['这是一本书', '我需要去超市购物', '明天是星期五']

# 创建标签
labels = [1] * len(positive_sentences) + [0] * len(normal_sentences)

# 创建词汇表
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(positive_sentences + normal_sentences)

# 将文本转换为数字表示
positive_sequences = tokenizer.texts_to_sequences(positive_sentences)
normal_sequences = tokenizer.texts_to_sequences(normal_sentences)

# 填充序列
max_length = 20
positive_sequences = keras.preprocessing.sequence.pad_sequences(positive_sequences, maxlen=max_length)
normal_sequences = keras.preprocessing.sequence.pad_sequences(normal_sequences, maxlen=max_length)

# 拆分训练和测试集
all_sequences = positive_sequences + normal_sequences
labels = tf.keras.utils.to_categorical(labels, num_classes=2)
train_data, test_data, train_labels, test_labels = train_test_split(all_sequences, labels, test_size=0.2, random_state=42)

# 构建神经网络模型
model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))







#参考下面


import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载IMDB数据集
num_words = 10000  # 仅保留训练集中最常见的单词
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# 为了使所有评论长度一致，我们将填充序列
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print("准确率:", accuracy)

