import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
# 下载SQuAD数据集
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# 加载SQuAD数据集
with open("train-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# 提取问题和答案对
questions = []
answers = []
for topic in squad_data["data"]:
    for paragraph in topic["paragraphs"]:
        for qa in paragraph["qas"]:
            questions.append(qa["question"])
            if qa["answers"]:
                answers.append(qa["answers"][0]["text"])
            else:
                answers.append("")  # 无答案的情况下添加空字符串

# 使用BERT预处理器
bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_preprocess_model = hub.load(bert_preprocess_url)
# 准备训练数据集
text_preprocessed = bert_preprocess_model(questions)
labels = np.ones(len(questions))  # 1 表示问题和答案对应关系为真
# 加载BERT模型
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_model = hub.load(bert_model_url)
# 构建分类模型
class BERTClassifier(tf.keras.Model):
    def __init__(self, bert_model):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert_model
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        bert_outputs = self.bert_model(inputs)
        pooled_output = bert_outputs["pooled_output"]
        return self.dense(pooled_output)
model = BERTClassifier(bert_model)
# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(text_preprocessed, labels, epochs=3, batch_size=32)











##########################
import tensorflow as tf
import tensorflow_hub as hub

# 加载BERT模型及其预处理器
# 加载BERT模型及其预处理器
bert_model_url = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4"
bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"

bert_preprocess_model = hub.load(bert_preprocess_url)
bert_model = hub.load(bert_model_url)

# 准备文本和问题
context = """法国是欧洲的一个国家，首都是巴黎。"""
question = "法国的首都是哪里？"

# 使用BERT预处理模型对文本和问题进行预处理
text_preprocessed = bert_preprocess_model([context, question])

# 获取BERT模型的输出
bert_outputs = bert_model(text_preprocessed)

# 提取答案
start_logits, end_logits = bert_outputs['sequence_output']
start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
end_idx = tf.argmax(end_logits, axis=1).numpy()[0]

# 从原始文本中提取答案
predicted_answer = context[start_idx:end_idx+1]
print("Predicted Answer:", predicted_answer)
# 使用BERT预处理器
bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_preprocess_model = hub.load(bert_preprocess_url)
import numpy as np
import json
# 准备训练数据集
text_preprocessed = bert_preprocess_model(questions)
labels = np.ones(len(questions))  # 1 表示问题和答案对应关系为真

# 加载BERT模型
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_model = hub.load(bert_model_url)

# 构建分类模型
input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
bert_outputs = bert_model(input_word_ids)
pooled_output = bert_outputs["pooled_output"]
output = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)

model = tf.keras.Model(inputs=input_word_ids, outputs=output)
