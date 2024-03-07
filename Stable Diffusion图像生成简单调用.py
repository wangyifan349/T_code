"""
pip install tensorflow
pip install keras-cv
pip install keras

#确保安装这些库
"""
import keras_cv
import keras
#import matplotlib.pyplot as plt
# 设置全局的混合精度策略为"mixed_float16"以提高性能
keras.mixed_precision.set_global_policy("mixed_float16")
# 创建Stable Diffusion模型的实例，启用XLA编译以进一步提高性能
model = keras_cv.models.StableDiffusion(
    img_width=512, img_height=512, jit_compile=True
)
##############################################
images = model.text_to_image("cat",batch_size=3)
#batch_size=3一次性生成3张图像
for i, image in enumerate(images):# 保存生成的图像到本地文件
    image.save(f"StableDiffusion_image_{i}.png")
