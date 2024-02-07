import tensorflow as tf
import numpy as np

# 加载并处理训练数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建并编译神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练神经网络模型
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 量化模型为8位
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# 保存量化的 TFLite 模型
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# 加载量化的 TFLite 模型
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

# 定义用于执行量化 TFLite 模型推理的辅助函数
def lite_model(images):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# 在数据集上进行推理并对比精度
num_eval_examples = 50
eval_dataset = ((image, label)  # TFLite 期望批处理大小为1
                for batch in test_images
                for (image, label) in zip(*batch))
count = 0
count_lite_tf_agree = 0
count_lite_correct = 0
for image, label in eval_dataset:
    probs_lite = lite_model(image[None, ...])[0]
    probs_tf = model(image[None, ...]).numpy()[0]
    y_lite = np.argmax(probs_lite)
    y_tf = np.argmax(probs_tf)
    y_true = label
    count += 1
    if y_lite == y_tf: count_lite_tf_agree += 1
    if y_lite == y_true: count_lite_correct += 1
    if count >= num_eval_examples: break
print("TFLite 模型在 %d 个示例中与原始模型一致 (%g%%)." %
      (count_lite_tf_agree, 100.0 * count_lite_tf_agree / count))
print("TFLite 模型在 %d 个示例中准确 (%g%%)." %
      (count_lite_correct, 100.0 * count_lite_correct / count))
