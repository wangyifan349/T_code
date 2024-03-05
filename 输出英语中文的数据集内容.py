import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import logging

# 设置 TensorFlow 日志记录级别以抑制不必要的消息
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 设置 NumPy 不使用科学计数法
np.set_printoptions(suppress=True)

# 定义数据集构建器
tmp_builder = tfds.builder('wmt19_translate/zh-en')

# 打印可用的子集
print(tmp_builder.subsets)

# 定义下载配置并下载数据
config = tfds.translate.wmt.WmtConfig(
    version=tfds.core.Version('0.0.7', experiments={tfds.core.Experiment.DUMMY: False}),
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v14"]
    }
)

# 下载和准备数据
builder = tfds.builder('wmt_translate', config=config)
builder.download_and_prepare(download_dir="C:\\tensorflow-datasets\wmt")

# 加载训练、验证和测试数据集
train_examples, val_examples, test_examples = builder.as_dataset(split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

# 函数：打印数据集中的一些样本
def print_samples(dataset, num_samples=10):
    c = 1
    for item in dataset.take(num_samples).as_numpy_iterator():
        print('-' * 10 + '样本 ' + str(c) + '-' * 10)
        print("英文:", item.get('en').decode())
        print("中文:", item.get('zh').decode())
        c += 1

# 打印训练数据集中的一些示例
print_samples(train_examples)

# 函数：提取并返回数据集中的样本
def extract_samples(dataset, num_samples=10):
    samples = []
    for item in dataset.take(num_samples).as_numpy_iterator():
        samples.append((item.get('zh').decode(), item.get('en').decode()))
    return samples

# 从训练数据集中提取样本
sample_data = extract_samples(train_examples)

# 函数：将提取的样本保存到文件中
def save_samples_to_file(samples, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample[0] + '\t' + sample[1] + '\n')

# 使用 save_samples_to_file 函数将提取的样本保存到文本文件 'extracted_samples.txt' 中
save_samples_to_file(sample_data, 'extracted_samples.txt')
