#这是一种捷径办法
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# 加载数据集
train_data, validation_data, test_data = tfds.load(
    name="wmt13_translate/es-en",
    split=("train[:80%]", "train[80%:90%]", "train[90%:]"),
    as_supervised=True
)

# 初始化BERT tokenizer
bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")

# 数据预处理函数
def preprocess(data):
    input_text = data[0]
    target_text = data[1]
    input_tokens = bert_preprocess(input_text)
    target_tokens = bert_preprocess(target_text)
    return {"input_word_ids": input_tokens["input_word_ids"], "input_mask": input_tokens["input_mask"], "input_type_ids": input_tokens["input_type_ids"]}, {"target_word_ids": target_tokens["input_word_ids"], "target_mask": target_tokens["input_mask"], "target_type_ids": target_tokens["input_type_ids"]}

# 准备训练数据
train_data = train_data.map(preprocess).shuffle(1000).padded_batch(32, padded_shapes=({"input_word_ids": [-1], "input_mask": [-1], "input_type_ids": [-1]}, {"target_word_ids": [-1], "target_mask": [-1], "target_type_ids": [-1]}))
validation_data = validation_data.map(preprocess).padded_batch(32, padded_shapes=({"input_word_ids": [-1], "input_mask": [-1], "input_type_ids": [-1]}, {"target_word_ids": [-1], "target_mask": [-1], "target_type_ids": [-1]}))
test_data = test_data.map(preprocess).padded_batch(32, padded_shapes=({"input_word_ids": [-1], "input_mask": [-1], "input_type_ids": [-1]}, {"target_word_ids": [-1], "target_mask": [-1], "target_type_ids": [-1]}))

# 构建模型
def build_model():
    input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_type_ids")

    target_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="target_word_ids")
    target_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="target_mask")
    target_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="target_type_ids")

    bert_inputs = {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
        "target_word_ids": target_word_ids,
        "target_mask": target_mask,
        "target_type_ids": target_type_ids,
    }

    bert_outputs = bert_encoder(bert_inputs)
    mlm_output = bert_outputs["pooled_output"]
    output = tf.keras.layers.Dense(2, activation='softmax')(mlm_output)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, target_word_ids, target_mask, target_type_ids], outputs=output)
    return model

model = build_model()

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 训练模型
model.fit(train_data, validation_data=validation_data, epochs=3)

# 评估模型
model.evaluate(test_data)

# 翻译函数
def translate(sentence):
    input_tokens = bert_preprocess(sentence)
    input_word_ids = input_tokens["input_word_ids"]
    input_mask = input_tokens["input_mask"]
    input_type_ids = input_tokens["input_type_ids"]

    # 使用训练好的模型进行翻译
    output = model.predict([input_word_ids, input_mask, input_type_ids, tf.zeros_like(input_word_ids), tf.zeros_like(input_mask), tf.zeros_like(input_type_ids)])
    
    # 获取预测结果
    predicted_class = tf.argmax(output, axis=-1).numpy()[0]
    
    # 将预测结果转换为文本
    translated_sentence = "English" if predicted_class == 1 else "Spanish"
    
    return translated_sentence

# 测试翻译函数
sentence = "Hola, ¿cómo estás?"
translated_sentence = translate(sentence)
print(f"Translated Sentence: {translated_sentence}")
