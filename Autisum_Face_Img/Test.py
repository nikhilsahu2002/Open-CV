import cv2
from mtcnn import MTCNN
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load and Preprocess Data
data_dir = r'E:\Open CV\Autisum_Face_Img\train'
autism_class = r'autistic'
no_autism_class = r'non_autistic'

autism_images = []
no_autism_images = []

for class_name in [autism_class, no_autism_class]:
    class_path = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (200, 200))
        if class_name == autism_class:
            autism_images.append(image)
        else:
            no_autism_images.append(image)

autism_images = np.array(autism_images)
no_autism_images = np.array(no_autism_images)

# Create Labels
autism_labels = np.ones(len(autism_images))
no_autism_labels = np.zeros(len(no_autism_images))

# Split Data for Training and Testing
X_train_autism, X_test_autism, y_train_autism, y_test_autism = train_test_split(
    autism_images, autism_labels, test_size=0.2, random_state=42)

X_train_no_autism, X_test_no_autism, y_train_no_autism, y_test_no_autism = train_test_split(
    no_autism_images, no_autism_labels, test_size=0.2, random_state=42)

# Combine Autism and Non-Autism Data for Testing
X_test = np.concatenate((X_test_autism, X_test_no_autism))
y_test = np.concatenate((y_test_autism, y_test_no_autism))

# Shuffle the Testing Data
shuffle_indices = np.arange(len(X_test))
np.random.shuffle(shuffle_indices)

X_test = X_test[shuffle_indices]
y_test = y_test[shuffle_indices]

# Build and Compile Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation and Training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use one of the class datasets for data augmentation
datagen.fit(X_train_autism)

batch_size = 16
epochs = 5

history = model.fit(
    datagen.flow(X_train_autism, y_train_autism, batch_size=batch_size),
    steps_per_epoch=len(X_train_autism) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# Evaluate and Predict
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Load MTCNN Face Detector
face_detector = MTCNN()

# Load and preprocess new image
new_image_path = r'E:\Open CV\Autisum_Face_Img\test\autistic\002.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Detect face using MTCNN
faces = face_detector.detect_faces(new_image)
if faces:
    x, y, width, height = faces[0]['box']
    face_img = new_image[y:y+height, x:x+width]
    face_img = cv2.resize(face_img, (200, 200))
    face_img = face_img / 255.0

    # Make Prediction
    prediction = model.predict(np.expand_dims(face_img, axis=0))
    if prediction[0][0] >= 0.5:
        print("Predicted: Autism")
    else:
        print("Predicted: No Autism")
else:
    print("No face detected in the image")
