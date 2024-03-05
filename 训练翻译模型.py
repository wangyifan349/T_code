import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 加载之前提取的样本数据
def load_samples_from_file(filename):
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            zh, en = line.strip().split('\t')
            samples.append((zh, en))
    return samples

# 加载样本数据
samples = load_samples_from_file('extracted_samples.txt')

# 分割数据为训练集和验证集
train_data, val_data = train_test_split(samples, test_size=0.1, random_state=42)

# 提取中文和英文句子
chinese_sentences, english_sentences = zip(*train_data)

# 建立中文词汇表
chinese_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
chinese_tokenizer.fit_on_texts(chinese_sentences)
chinese_vocab_size = len(chinese_tokenizer.word_index) + 1

# 建立英文词汇表
english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
english_tokenizer.fit_on_texts(english_sentences)
english_vocab_size = len(english_tokenizer.word_index) + 1

# 将文本转换为序列
chinese_sequences = chinese_tokenizer.texts_to_sequences(chinese_sentences)
english_sequences = english_tokenizer.texts_to_sequences(english_sentences)

# 填充序列，使其长度一致
max_sequence_length = 50
chinese_padded = tf.keras.preprocessing.sequence.pad_sequences(chinese_sequences, maxlen=max_sequence_length, padding='post')
english_padded = tf.keras.preprocessing.sequence.pad_sequences(english_sequences, maxlen=max_sequence_length, padding='post')

# 构建模型
embedding_dim = 256
units = 1024
batch_size = 64

# 编码器
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(chinese_vocab_size, embedding_dim, units, batch_size)

# 注意力层
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

attention_layer = BahdanauAttention(units)

# 解码器
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(english_vocab_size, embedding_dim, units, batch_size)

# 优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 损失函数
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# 训练步骤
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([english_tokenizer.word_index['<start>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

# 建立检查点
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# 训练
EPOCHS = 10

for epoch in range(EPOCHS):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # 每 2 个周期（epoch）保存（检查点）一次模型
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
