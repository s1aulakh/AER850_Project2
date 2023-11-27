# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Data Processing
# Define input image shape
input_shape = (100, 100, 3)

# Define data directories
train_data_dir = 'C:/Users/simra/Downloads/Data/train'
validation_data_dir = 'C:/Users/simra/Downloads/Data/validation'


# Set up data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)

# Set up data augmentation for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 2: Neural Network Architecture Design
# Model 1: Higher accuracy model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Model 2: Lower accuracy model (commented out)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(4, activation='softmax'))

# Step 3: Hyperparameter Analysis
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Step 4: Model Evaluation
epochs = 45
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Plotting accuracy and loss
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 5: Model Testing
# Load and preprocess test images
from tensorflow.keras.preprocessing import image
import numpy as np

test_image_path1 = 'C:/Users/simra/Downloads/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'
test_image1 = image.load_img(test_image_path1, target_size=input_shape[:2])
test_image_array1 = image.img_to_array(test_image1)
test_image_array1 = np.expand_dims(test_image_array1, axis=0)
test_image_array1 /= 255.0

test_image_path2 = 'C:/Users/simra/Downloads/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'
test_image2 = image.load_img(test_image_path2, target_size=input_shape[:2])
test_image_array2 = image.img_to_array(test_image2)
test_image_array2 = np.expand_dims(test_image_array2, axis=0)
test_image_array2 /= 255.0
# Model prediction
predictions1 = model.predict(test_image_array1)
predictions2 = model.predict(test_image_array2)

# Find the class with the maximum probability
predicted_class1 = np.argmax(predictions1)
predicted_class2 = np.argmax(predictions2)

#Map the class index to the class label1
class_labels1 = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'No Crack'}
predicted_label1 = class_labels1[predicted_class1]

#Display the prediction on the image
plt.imshow(test_image1)
plt.axis('off')

#Display percentages on the image
for i, label in enumerate(class_labels1.values()):
    percentage1 = predictions1[0][i] * 100
    plt.text(10, 10 + i * 20, f'{label}: {percentage1:.2f}%', color='white', backgroundcolor='black', fontsize=8)

plt.show()

# Map the class index to the class label2
class_labels2 = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'No Crack'}
predicted_label2 = class_labels2[predicted_class2]

# Display the prediction on the image
plt.imshow(test_image2)
plt.axis('off')

# Display percentages on the image
for i, label in enumerate(class_labels2.values()):
    percentage2 = predictions2[0][i] * 100
    plt.text(10, 10 + i * 20, f'{label}: {percentage2:.2f}%', color='white', backgroundcolor='black', fontsize=8)

plt.show()