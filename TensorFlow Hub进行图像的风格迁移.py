import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# 打印当前环境的TensorFlow版本、TensorFlow Hub版本、是否启用Eager模式、是否有GPU可用
print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# 定义图像裁剪函数，用于将图像裁剪为中心的正方形
def crop_center(image):
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

# 使用functools.lru_cache装饰器缓存加载的图像，避免重复加载
@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  # 根据URL下载图像，并缓存到本地
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # 加载图像，转换为float32类型的numpy数组，并添加批次维度
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)  # 裁剪图像
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)  # 调整图像大小
  return img

# 显示一组图像及其标题的函数
def show_n(images, titles=('',)):
  # 计算要显示的图像数量
  n = len(images)
  
  # 获取每个图像的宽度，用于后续计算显示大小
  image_sizes = [image.shape[1] for image in images]
  
  # 计算每个图像显示时的宽度，这里的计算基于第一个图像的宽度，并按比例缩放
  w = (image_sizes[0] * 6) // 320
  
  # 创建一个新的matplotlib图形窗口，设置窗口大小
  plt.figure(figsize=(w * n, w))
  
  # 使用gridspec创建一个网格布局，这里创建了1行n列的布局，每列的宽度比由image_sizes决定
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  
  # 遍历每个图像及其标题，进行显示
  for i in range(n):
    # 在网格布局中为当前图像创建一个子图
    plt.subplot(gs[i])
    # 显示图像，这里的images[i][0]表示取出第i个图像的第一个元素（因为图像可能有批次维度）
    plt.imshow(images[i][0], aspect='equal')
    # 关闭坐标轴显示
    plt.axis('off')
    # 设置图像的标题，如果提供了标题列表且当前图像有对应标题，则使用对应标题，否则不显示标题
    plt.title(titles[i] if len(titles) > i else '')
  # 显示所有图像和标题
  plt.show()

# 加载示例内容图像和风格图像
content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'
style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
output_image_size = 384  # 设置输出图像的大小

# 设置内容图像和风格图像的大小
content_img_size = (output_image_size, output_image_size)
style_img_size = (256, 256)  # 推荐将风格图像大小设置为256

# 加载并处理图像
content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')  # 对风格图像应用平均池化

# 显示内容图像和风格图像
show_n([content_image, style_image], ['Content image', 'Style image'])

# 定义更多的内容图像和风格图像URL
content_urls = dict(
  sea_turtle='https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg',
  tuebingen='https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg',
  grace_hopper='https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg',
  )
style_urls = dict(
  kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
  kandinsky_composition_7='https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
  hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
  van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
  turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
  munch_scream='https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
  picasso_demoiselles_avignon='https://upload.wikimedia.org/wikipedia/en/4/4c/Les_Demoiselles_d%27Avignon.jpg',
  picasso_violin='https://upload.wikimedia.org/wikipedia/en/3/3c/Pablo_Picasso%2C_1911-12%2C_Violon_%28Violin%29%2C_oil_on_canvas%2C_Kr%C3%B6ller-M%C3%BCller_Museum%2C_Otterlo%2C_Netherlands.jpg',
  picasso_bottle_of_rum='https://upload.wikimedia.org/wikipedia/en/7/7f/Pablo_Picasso%2C_1911%2C_Still_Life_with_a_Bottle_of_Rum%2C_oil_on_canvas%2C_61.3_x_50.5_cm%2C_Metropolitan_Museum_of_Art%2C_New_York.jpg',
  fire='https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg',
  derkovits_woman_head='https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg',
  amadeo_style_life='https://upload.wikimedia.org/wikipedia/commons/8/8e/Untitled_%28Still_life%29_%281913%29_-_Amadeo_Souza-Cardoso_%281887-1918%29_%2817385824283%29.jpg',
  derkovtis_talig='https://upload.wikimedia.org/wikipedia/commons/3/37/Derkovits_Gyula_Talig%C3%A1s_1920.jpg',
  amadeo_cardoso='https://upload.wikimedia.org/wikipedia/commons/7/7d/Amadeo_de_Souza-Cardoso%2C_1915_-_Landscape_with_black_figure.jpg'
)
# 加载并处理更多的内容图像和风格图像
content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}
style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}
