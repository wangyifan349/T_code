from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# 加载预训练的 BERT 模型和 tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备数据集（示例）
questions = ['Who invented the World Wide Web?', 'What is the capital of France?']
texts = [
    "The World Wide Web was invented by Tim Berners-Lee in 1989.",
    "Paris is the capital of France."
]
answers = ["Tim Berners-Lee", "Paris"]

# 对每个问题和文本段落进行分词，并转换成模型可接受的输入格式
inputs = tokenizer(questions, texts, return_tensors='pt', padding=True, truncation=True)

# 将数据传递给模型，得到答案的起始和结束位置的 logits
outputs = model(**inputs)

# 提取答案起始和结束位置的预测
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 打印预测的答案
for i, (start_logit, end_logit) in enumerate(zip(start_logits, end_logits)):
    start_index = torch.argmax(start_logit)
    end_index = torch.argmax(end_logit)
    answer_tokens = inputs['input_ids'][i][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)
    print("Question:", questions[i])
    print("Predicted Answer:", answer)
    print("True Answer:", answers[i])
    print()
