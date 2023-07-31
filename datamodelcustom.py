import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Define the path to your data directory
data_directory = r'E:\Open CV\your_dataset_directory'

# Function to load and preprocess images
def load_and_preprocess_image(file_path, target_size=(128, 128)):
    img = Image.open(file_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array

# Load and preprocess the dataset
X = []
y = []
for class_name in os.listdir(data_directory):
    class_dir = os.path.join(data_directory, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        X.append(load_and_preprocess_image(file_path))
        y.append(1 if class_name == 'autism' else 0)

X = np.array(X)
y = np.array(y)

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Fit the model
batch_size = 32
num_epochs = 50

model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size, epochs=num_epochs, validation_data=(X_valid, y_valid))

# Load an MRI image for prediction
img_path = r'E:\Open CV\Screenshot (317).png'
img_array = load_and_preprocess_image(img_path)
img_array = np.expand_dims(img_array, axis=0)

# Make the prediction
prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print("The MRI image does not have Autism.")
else:
    print("The MRI image has Autism.")
