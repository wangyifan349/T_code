%%capture
#@title Setup Environment
# Install the latest Tensorflow version.
!pip install -q "tensorflow-text==2.11.*"
!pip install -q simpleneighbors[annoy]
!pip install -q nltk
!pip install -q tqdm

# 导入必要的库
import json
import nltk  # 用于文本处理
import os
import pprint
import random
import simpleneighbors  # 用于建立简单的邻近搜索索引
import urllib
from IPython.display import HTML, display
from tqdm.notebook import tqdm  # 用于显示进度条

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

nltk.download('punkt')  # 下载punkt分词器

# 定义函数下载SQuAD数据集
def download_squad(url):
  return json.load(urllib.request.urlopen(url))

# 从SQuAD数据集JSON格式中提取句子
def extract_sentences_from_squad_json(squad):
  all_sentences = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      sentences = nltk.tokenize.sent_tokenize(paragraph['context'])
      all_sentences.extend(zip(sentences, [paragraph['context']] * len(sentences)))
  return list(set(all_sentences))  # 移除重复项

# 从SQuAD数据集JSON格式中提取问题及其答案
def extract_questions_from_squad_json(squad):
  questions = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      for qas in paragraph['qas']:
        if qas['answers']:
          questions.append((qas['question'], qas['answers'][0]['text']))
  return list(set(questions))

# 以高亮显示的方式输出文本和答案
def output_with_highlight(text, highlight):
  output = "<li> "
  i = text.find(highlight)
  while True:
    if i == -1:
      output += text
      break
    output += text[0:i]
    output += '<b>'+text[i:i+len(highlight)]+'</b>'
    text = text[i+len(highlight):]
    i = text.find(highlight)
  return output + "</li>\n"

# 显示与查询文本最接近的句子
def display_nearest_neighbors(query_text, answer_text=None):
  query_embedding = model.signatures['question_encoder'](tf.constant([query_text]))['outputs'][0]
  search_results = index.nearest(query_embedding, n=num_results)

  if answer_text:
    result_md = '''
    <p>Random Question from SQuAD:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    <p>Answer:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % (query_text , answer_text)
  else:
    result_md = '''
    <p>Question:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % query_text

  result_md += '''
    <p>Retrieved sentences :
    <ol>
  '''

  if answer_text:
    for s in search_results:
      result_md += output_with_highlight(s, answer_text)
  else:
    for s in search_results:
      result_md += '<li>' + s + '</li>\n'

  result_md += "</ol>"
  display(HTML(result_md))


# 选择SQuAD数据集的URL，可以根据需要更改为不同的版本
squad_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'

# 下载SQuAD数据集
squad_json = download_squad(squad_url)

# 从下载的JSON数据中提取句子
sentences = extract_sentences_from_squad_json(squad_json)

# 从下载的JSON数据中提取问题及其答案
questions = extract_questions_from_squad_json(squad_json)

# 打印提取的句子和问题的数量，以及数据集的URL
print("%s sentences, %s questions extracted from SQuAD %s" % (len(sentences), len(questions), squad_url))

# 打印一个示例句子及其上下文
print("\nExample sentence and context:\n")
sentence = random.choice(sentences)  # 随机选择一个句子
print("sentence:\n")
pprint.pprint(sentence[0])  # 打印句子
print("\ncontext:\n")
pprint.pprint(sentence[1])  # 打印句子所在的上下文
#@title Load model from tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
model = hub.load(module_url)

# 设置批处理大小
batch_size = 100

# 对第一个句子和其上下文计算嵌入向量，主要用于获取嵌入向量的维度
encodings = model.signatures['response_encoder'](
  input=tf.constant([sentences[0][0]]),
  context=tf.constant([sentences[0][1]]))

# 使用获取到的嵌入向量的维度初始化一个简单的邻近搜索索引
index = simpleneighbors.SimpleNeighbors(
    len(encodings['outputs'][0]), metric='angular')

# 开始处理并计算所有句子的嵌入向量
print('Computing embeddings for %s sentences' % len(sentences))

# 将句子列表分成多个批次，每个批次包含batch_size数量的句子
slices = zip(*(iter(sentences),) * batch_size)
num_batches = int(len(sentences) / batch_size)

# 使用tqdm显示处理进度
for s in tqdm(slices, total=num_batches):
  # 分别提取当前批次中所有句子及其对应的上下文
  response_batch = list([r for r, c in s])
  context_batch = list([c for r, c in s])
  # 对当前批次的所有句子和上下文计算嵌入向量
  encodings = model.signatures['response_encoder'](
    input=tf.constant(response_batch),
    context=tf.constant(context_batch)
  )
  # 将当前批次的句子及其嵌入向量添加到索引中
  for batch_index, batch in enumerate(response_batch):
    index.add_one(batch, encodings['outputs'][batch_index])

# 构建索引，以便进行快速的近似最近邻搜索
index.build()
print('simpleneighbors index for %s sentences built.' % len(sentences))

num_results = 10

query = random.choice(questions)
display_nearest_neighbors(query[0], query[1])











