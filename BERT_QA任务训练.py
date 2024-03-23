import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from datasets import load_dataset
# 加载SQuAD数据集
squad = load_dataset("squad")

# 初始化BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def prepare_data(example):
    inputs = tokenizer(example["question"], example["context"], return_tensors="tf")
    if "answers" in example and "answer_start" in example["answers"] and "answer_end" in example["answers"]:
        start_positions = tf.convert_to_tensor([example["answers"]["answer_start"][0]])
        end_positions = tf.convert_to_tensor([example["answers"]["answer_end"][0]])
    else:
        # If "answers" key is not present or if "answer_start" and "answer_end" are missing, set start and end positions to 0
        start_positions = tf.convert_to_tensor([0])
        end_positions = tf.convert_to_tensor([0])
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# 数据预处理
train_data = squad["train"].map(prepare_data).shuffle(buffer_size=1000).batch(batch_size)

# 加载 TensorFlow Hub 中的 BERT 模型
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
bert_layer = hub.KerasLayer(bert_model_url, trainable=True)

# 定义模型结构
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
start_positions = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="start_positions")
end_positions = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="end_positions")

outputs = bert_layer({"input_ids": input_ids, "attention_mask": attention_mask})
sequence_output = outputs["sequence_output"]
start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)

start_logits = tf.keras.layers.Flatten()(start_logits)
end_logits = tf.keras.layers.Flatten()(end_logits)

model = tf.keras.models.Model(inputs=[input_ids, attention_mask, start_positions, end_positions], outputs=[start_logits, end_logits])

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 编译模型
model.compile(optimizer=optimizer, loss=[loss_fn, loss_fn])

# 训练模型
model.fit(train_data, epochs=num_epochs)





# 保存模型
model.save("bert_qa_model")

# 加载模型
loaded_model = tf.keras.models.load_model("bert_qa_model")

# 定义函数以对新问题进行预测
def predict_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="tf", truncation=True)
    start_logits, end_logits = loaded_model.predict([inputs["input_ids"], inputs["attention_mask"]])
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer

# 询问问题
question = "What is the capital of France?"
context = "The capital of France is Paris."
predicted_answer = predict_answer(question, context)
print("Predicted answer:", predicted_answer)



