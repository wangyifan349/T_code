import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer, TFBertForQuestionAnswering

# 加载 SQuAD 2.0 数据集
squad_train, squad_validation = tfds.load('squad/2.0', split=['train', 'validation'], shuffle_files=True)

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将数据集预处理为模型可接受的格式
def preprocess_data(example):
    question = example['question']
    context = example['context']
    answer_text = example['answers']['text'][0]
    start_char_idx = example['answers']['answer_start'][0]
    end_char_idx = start_char_idx + len(answer_text)

    # 对问题和上下文进行编码
    input_dict = tokenizer(question, context, padding='max_length', max_length=384, truncation=True, return_tensors='tf')
    
    # 获取答案在 tokenized 上下文中的起始和结束位置
    start_token_idx = tokenizer.convert_chars_to_token_indices(start_char_idx)
    end_token_idx = tokenizer.convert_chars_to_token_indices(end_char_idx - 1)

    # 创建答案掩码
    input_dict['start_positions'] = [start_token_idx] if start_token_idx is not None else [-1]
    input_dict['end_positions'] = [end_token_idx] if end_token_idx is not None else [-1]
    
    return input_dict

# 对数据集进行预处理
train_dataset = squad_train.map(preprocess_data).shuffle(1000).batch(8)
validation_dataset = squad_validation.map(preprocess_data).batch(8)

# 加载预训练的 BERT 模型
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# 训练模型
model.fit(train_dataset, epochs=2, validation_data=validation_dataset)

# 在测试集上进行评估
results = model.evaluate(validation_dataset)
print("Validation loss:", results[0])
print("Validation accuracy:", results[1])

# 使用模型回答问题
def answer_question(question, context):
    input_dict = tokenizer(question, context, padding='max_length', max_length=384, truncation=True, return_tensors='tf')
    start_logits, end_logits = model(input_dict)
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0] + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_dict['input_ids'][0][start_idx:end_idx]))
    return answer

# 例子：给定问题和上下文，使用模型回答问题
question = "What is SQuAD 2.0?"
context = "SQuAD 2.0 is a dataset for question answering. It is an improved version of SQuAD 1.1."

answer = answer_question(question, context)
print("Question:", question)
print("Answer:", answer)
