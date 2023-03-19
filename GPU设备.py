import tensorflow as tf

# 获取所有可用的GPU设备列表
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 遍历每个GPU设备以启用动态内存分配（避免一次性分配全部内存）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 选择尽可能多的GPU设备进行计算
        num_gpus = len(gpus)
        tf.config.set_visible_devices(gpus[:num_gpus], 'GPU')

        # 测试设备放置情况，并使用一个简单的操作测试GPU设备是否可用
        with tf.device('/device:GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c = tf.matmul(a, b)
        # 打印所使用的GPU的名称
        print("所用GPU：", tf.test.gpu_device_name())
    except RuntimeError as e:
        print(e)
else:
    print("没有可用的GPU设备。")
