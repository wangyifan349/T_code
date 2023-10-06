import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Download and load the TF Flowers dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.tf_flowers.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Split the dataset into training and testing sets (80% training, 20% testing)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Create a data generator for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_datagen.fit(train_images)

# Build a convolutional neural network (CNN) model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes in TF Flowers dataset
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=32),
                    validation_data=(val_images, val_labels),
                    epochs=10)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the trained model to an .h5 file
model.save('flower_recognition_model.h5')
######################
# Load the trained model
model = load_model('flower_recognition_model.h5')

# Load and preprocess the image you want to classify
img_path = 'path_to_your_image.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values

# Make predictions using the model
predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)

# Map the class index to the corresponding flower label
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
flower_label = class_labels[predicted_class]

print(f'The flower in the image is: {flower_label}')
