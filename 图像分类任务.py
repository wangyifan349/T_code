# 引入 Keras 模块
from keras.applications import ResNet50, VGG16, InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# 载入预训练的 ResNet50 模型
resnet_model = ResNet50(weights='imagenet')

# 载入预训练的 VGG16 模型
vgg_model = VGG16(weights='imagenet')

# 载入预训练的 InceptionV3 模型
inception_model = InceptionV3(weights='imagenet')

# 加载并预处理图像
img_path = 'your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 使用 ResNet50 进行图像分类
predictions_resnet = resnet_model.predict(x)
print('ResNet50 Predictions:')
print(decode_predictions(predictions_resnet, top=3)[0])

# 使用 VGG16 进行图像分类
predictions_vgg = vgg_model.predict(x)
print('VGG16 Predictions:')
print(decode_predictions(predictions_vgg, top=3)[0])

# 使用 InceptionV3 进行图像分类
predictions_inception = inception_model.predict(x)
print('InceptionV3 Predictions:')
print(decode_predictions(predictions_inception, top=3)[0])
