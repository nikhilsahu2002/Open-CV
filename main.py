import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from PIL import Image

# Define the target size for resizing images
target_size = (128, 128)

# Create your custom model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*target_size, 3)))
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
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Define the path to your data directory
data_directory = r'E:\Open CV\Autism Dataset'

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

batch_size = 32

# Load and preprocess the training data
train_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=target_size,  # Resize all images to the target size
    batch_size=batch_size,
    class_mode='binary'
)

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model with EarlyStopping callback
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    callbacks=[early_stopping]  # Add the EarlyStopping callback
)

# Function to load and preprocess a single image for prediction
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)

    # Ensure the image is in RGB mode (convert if needed)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.
    return img_array

# Load an image for prediction
img_path = r'E:\Open CV\Screenshot (320).png'
img_array = load_and_preprocess_image(img_path)

# Make the prediction
prediction = model.predict(img_array)
print("percentage of Prediction Is ", prediction)
if prediction[0] < 0.5:
    print("The MRI image does not have Autism.")
else:
    print("The MRI image has Autism.")
