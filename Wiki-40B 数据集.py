from datasets import load_dataset
from datasets import list_datasets

# 获取所有可用数据集的列表
datasets_list = list_datasets()

# 打印数据集列表
for dataset_name in datasets_list:
    print(dataset_name)



# 加载 Wiki-40B 数据集
dataset = load_dataset('wiki40b', split='train[:10]')

# 查看前几个样本
for example in dataset:
    print(example)
