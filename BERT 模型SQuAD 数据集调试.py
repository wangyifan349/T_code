from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from datasets import load_dataset

# 加载 SQuAD 数据集
squad_dataset = load_dataset('squad')

# 获取训练集
train_data = squad_dataset['train']

# 加载预训练的 BERT 模型和 tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 提取问题和相应的段落
questions = [example['question'] for example in train_data]  # 提取问题
paragraphs = [example['context'] for example in train_data]  # 提取段落
answers = [example['answers']['text'][0] for example in train_data]  # 获取第一个答案

# 对每个问题和文本段落进行分词，并转换成模型可接受的输入格式
inputs = tokenizer(questions, paragraphs, return_tensors='pt', padding=True, truncation=True)

# 将数据传递给模型，得到答案的起始和结束位置的 logits
outputs = model(**inputs)

# 提取答案起始和结束位置的预测
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 打印预测的答案
for i, (start_logit, end_logit) in enumerate(zip(start_logits, end_logits)):
    start_index = torch.argmax(start_logit)  # 获取起始位置的索引
    end_index = torch.argmax(end_logit)  # 获取结束位置的索引
    answer_tokens = inputs['input_ids'][i][start_index : end_index + 1]  # 从输入中提取答案的 token
    answer = tokenizer.decode(answer_tokens)  # 将 token 转换为文本
    print("Question:", questions[i])  # 打印问题
    print("Predicted Answer:", answer)  # 打印预测的答案
    print("True Answer:", answers[i])  # 打印真实答案
    print()
