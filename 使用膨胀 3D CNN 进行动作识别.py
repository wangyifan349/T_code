"""
pip install tensorflow tensorflow-hub opencv-python-headless numpy
pip install -q imageio
pip install -q opencv-python
pip install -q git+https://github.com/tensorflow/docs

"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2  # 用于视频处理

# 假设你已经有了处理 UCF101 数据集的函数
# list_ucf_videos, fetch_ucf_video, load_video

# 获取 UCF101 数据集中的视频列表
ucf_videos = list_ucf_videos()

# 初始化一个字典来存储每个类别及其对应的视频
categories = {}
# 遍历视频列表
for video in ucf_videos:
    category = video[2:-12]
    if category not in categories:# 如果这个类别还没有被记录在字典中，则添加进去
        categories[category] = []
    categories[category].append(video)    # 将当前视频添加到其对应类别的列表中

# 打印总共找到的视频数量和类别数量
print("Found %d videos in %d categories." % (len(ucf_videos), len(categories)))

# 遍历每个类别及其对应的视频列表
for category, sequences in categories.items():
    # 从每个类别的视频列表中取出前两个视频（如果有的话）作为示例
    summary = ", ".join(sequences[:2])
    # 打印类别名称、该类别下视频的数量以及示例视频名称
    print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))







# 加载 I3D 模型
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

# 加载标签
with open("path/to/kinetics-400_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def predict(sample_video):
    # 将视频转换为模型所需的形状
    # 假设 sample_video 已经是正确的形状 (num_frames, height, width, 3)
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)

    print("Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")

# 示例：加载和预处理视频
# 这里假设 fetch_ucf_video 和 load_video 已经完成了视频的加载和预处理
video_path = fetch_ucf_video("v_CricketShot_g04_c02.avi")
sample_video = load_video(video_path)

# 预测视频中的动作
predict(sample_video)


# curl -O https://upload.wikimedia.org/wikipedia/commons/8/86/End_of_a_jam.ogv


new_video_path = "End_of_a_jam.ogv"# 加载和预处理新的视频
new_sample_video = load_video(new_video_path)

# 预测新视频中的动作
predict(new_sample_video)
