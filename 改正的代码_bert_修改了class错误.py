import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

# 下载SQuAD数据集
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# 加载SQuAD数据集
with open("train-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# 提取问题、上下文和答案
contexts = []
questions = []
answers = []
for topic in squad_data["data"]:
    for paragraph in topic["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            questions.append(question)
            contexts.append(context)
            if qa["answers"]:
                answer_text = qa["answers"][0]["text"]
                answers.append(answer_text)
            else:
                answers.append("")  # 无答案的情况下添加空字符串

# 使用BERT预处理器
bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_preprocess_model = hub.load(bert_preprocess_url)

# 准备训练数据集
text_preprocessed = bert_preprocess_model([contexts, questions])
start_positions = []  # 起始位置列表
end_positions = []  # 结束位置列表
for answer, context in zip(answers, contexts):
    start_index = context.find(answer)
    end_index = start_index + len(answer)
    start_positions.append(start_index)
    end_positions.append(end_index)

# 加载BERT模型
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_model = hub.load(bert_model_url)

# 构建问答模型
class BERTQA(tf.keras.Model):
    def __init__(self, bert):
        super(BERTQA, self).__init__()
        self.bert = bert
        self.start_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.end_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        bert_outputs = self.bert(inputs)
        pooled_output = bert_outputs["pooled_output"]
        start_logits = self.start_output(pooled_output)
        end_logits = self.end_output(pooled_output)
        return start_logits, end_logits

model = BERTQA(bert_model)

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def qa_loss(start_positions, end_positions, start_logits, end_logits):
    start_loss = loss_object(start_positions, start_logits)
    end_loss = loss_object(end_positions, end_logits)
    total_loss = start_loss + end_loss
    return total_loss

@tf.function
def train_step(inputs, start_positions, end_positions):
    with tf.GradientTape() as tape:
        start_logits, end_logits = model(inputs, training=True)
        loss = qa_loss(start_positions, end_positions, start_logits, end_logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 准备数据集
dataset = tf.data.Dataset.from_tensor_slices((text_preprocessed, start_positions, end_positions))
dataset = dataset.shuffle(len(text_preprocessed)).batch(batch_size)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataset:
        inputs, start_positions_batch, end_positions_batch = batch
        loss = train_step(inputs, start_positions_batch, end_positions_batch)
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')
