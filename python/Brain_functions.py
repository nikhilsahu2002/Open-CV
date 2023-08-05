# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread(r'E:\Open CV\Alzheimer_s Dataset\test\MildDemented\27 (2).jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Preprocess the image (if needed)
# # Apply any necessary preprocessing steps, such as denoising or smoothing.

# # Calculate LBP features
# radius = 1
# neighbors = 8
# lbp = np.zeros_like(gray)
# for row in range(1, gray.shape[0] - 1):
#     for col in range(1, gray.shape[1] - 1):
#         center_pixel = gray[row, col]
#         pattern = 0
#         for i in range(neighbors):
#             angle = 2 * np.pi * i / neighbors
#             x = int(np.round(row + radius * np.cos(angle)))
#             y = int(np.round(col - radius * np.sin(angle)))
#             neighbor_pixel = gray[x, y]
#             pattern <<= 1
#             pattern |= 1 if neighbor_pixel >= center_pixel else 0
#         lbp[row, col] = 1 if pattern >= (1 << neighbors - 1) else -1

# # Calculate histogram of LBP features
# histogram, _ = np.histogram(lbp.flatten(), bins=256, range=(-1, 1))

# # Normalize the histogram
# normalized_histogram = histogram.astype('float32') / histogram.sum()

# # Print or use the extracted features
# print("LBP Histogram:")
# num_cols = 20  # Number of columns to display
# for i in range(0, len(normalized_histogram), num_cols):
#     row_values = normalized_histogram[i:i+num_cols]
#     row_string = " ".join(["  1" if value > 0 else "-1" for value in row_values])
#     print(row_string)

# #---------------------------------------------------------  Euclidean distance-----------------------------------------------

# import cv2
# import numpy as np
# from scipy.spatial import distance

# # Function to calculate LBP histogram
# def calculate_lbp_histogram(image):
#     radius = 1
#     neighbors = 8
#     lbp = np.zeros_like(image)
#     for row in range(1, image.shape[0] - 1):
#         for col in range(1, image.shape[1] - 1):
#             center_pixel = image[row, col]
#             pattern = 0
#             for i in range(neighbors):
#                 angle = 2 * np.pi * i / neighbors
#                 x = int(np.round(row + radius * np.cos(angle)))
#                 y = int(np.round(col - radius * np.sin(angle)))
#                 neighbor_pixel = image[x, y]
#                 pattern <<= 1
#                 pattern |= 1 if neighbor_pixel >= center_pixel else 0
#             lbp[row, col] = 1 if pattern >= (1 << neighbors - 1) else -1

#     histogram, _ = np.histogram(lbp.flatten(), bins=256, range=(-1, 1))
#     normalized_histogram = histogram.astype('float32') / histogram.sum()
#     return normalized_histogram

# # Load the images
# image1 = cv2.imread(r'E:\Open CV\Alzheimer_s Dataset\test\ModerateDemented\27 (2).jpg', 0)
# image2 = cv2.imread(r'E:\Open CV\Alzheimer_s Dataset\test\MildDemented\27 (2).jpg', 0)

# # Calculate LBP histograms
# histogram1 = calculate_lbp_histogram(image1)
# histogram2 = calculate_lbp_histogram(image2)

# # Compute the Euclidean distance
# euclidean_distance = distance.euclidean(histogram1, histogram2)

# # Print the Euclidean distance
# print("Euclidean Distance:", euclidean_distance)

# # Define a similarity threshold
# threshold = 0.1

# # Compare the Euclidean distance with the threshold
# if euclidean_distance < threshold:
#     print("The images are similar.")
# else:
#     print("The images are dissimilar.")

# --------------- Applying  CNN Algorith Using Tensorflow -------------------------------------

# import cv2
# import numpy as np
# from scipy.spatial import distance
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape

# # Function to calculate LBP histogram
# def calculate_lbp_histogram(image):
#     radius = 1
#     neighbors = 8
#     lbp = np.zeros_like(image)
#     for row in range(1, image.shape[0] - 1):
#         for col in range(1, image.shape[1] - 1):
#             center_pixel = image[row, col]
#             pattern = 0
#             for i in range(neighbors):
#                 angle = 2 * np.pi * i / neighbors
#                 x = int(np.round(row + radius * np.cos(angle)))
#                 y = int(np.round(col - radius * np.sin(angle)))
#                 neighbor_pixel = image[x, y]
#                 pattern <<= 1
#                 pattern |= 1 if neighbor_pixel >= center_pixel else 0
#             lbp[row, col] = 1 if pattern >= (1 << neighbors - 1) else -1

#     histogram, _ = np.histogram(lbp.flatten(), bins=256, range=(-1, 1))
#     normalized_histogram = histogram.astype('float32') / histogram.sum()
#     return normalized_histogram

# # Load the images
# image1 = cv2.imread(r'E:\Open CV\Asuna_SAO.png', 0)
# image2 = cv2.imread(r'E:\Open CV\2.png', 0)

# # Resize the images to smaller size
# resized_image1 = cv2.resize(image1, (100, 100))
# resized_image2 = cv2.resize(image2, (100, 100))


# # Calculate LBP histograms
# histogram1 = calculate_lbp_histogram(resized_image1)
# histogram2 = calculate_lbp_histogram(resized_image2)

# # Reshape the histograms for input to the CNN
# histogram1 = histogram1.reshape(1, 256, 1, 1)
# histogram2 = histogram2.reshape(1, 256, 1, 1)

# # Create a CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(256, 1, 1)))
# model.add(Conv2D(32, (3, 1), activation='relu'))
# model.add(MaxPooling2D((2, 1)))
# model.add(Conv2D(64, (3, 1), activation='relu'))
# model.add(Conv2D(64, (3, 1), activation='relu'))
# model.add(MaxPooling2D((2, 1)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Compile and train the CNN model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # model.fit(np.concatenate([histogram1, histogram2]), np.array([0, 1]), epochs=10)
# model.fit(np.concatenate([histogram1, histogram2]), np.array([-1, 1]), epochs=10)


# # Predict the similarity using the trained model
# prediction = model.predict(np.concatenate([histogram1, histogram2]))
# # if prediction[0] > 0.5:
# if prediction[0] > 0:
#     print("The images are similar.")
# else:
#     print("The images are dissimilar.")
# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from skimage.feature import hog
# from sklearn.metrics import accuracy_score

# # Function to extract HOG features
# def extract_hog_features(image):
#     features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
#     return features

# # Function to calculate LBP histogram
# def calculate_lbp_histogram(image):
#     radius = 1
#     neighbors = 8
#     lbp = np.zeros_like(image)
#     for row in range(1, image.shape[0] - 1):
#         for col in range(1, image.shape[1] - 1):
#             center_pixel = image[row, col]
#             pattern = 0
#             for i in range(neighbors):
#                 angle = 2 * np.pi * i / neighbors
#                 x = int(np.round(row + radius * np.cos(angle)))
#                 y = int(np.round(col - radius * np.sin(angle)))
#                 neighbor_pixel = image[x, y]
#                 pattern <<= 1
#                 pattern |= 1 if neighbor_pixel >= center_pixel else 0
#             lbp[row, col] = 1 if pattern >= (1 << neighbors - 1) else -1

#     histogram, _ = np.histogram(lbp.flatten(), bins=256, range=(-1, 1))
#     normalized_histogram = histogram.astype('float32') / histogram.sum()
#     return normalized_histogram

# # Load the images
# reference_image = cv2.imread(r'E:\Open CV\Asuna_with_Yui_Biprobe.png', 0)
# test_image = cv2.imread(r'E:\Open CV\imag2.png', 0)

# # Resize the images to smaller size
# resized_reference_image = cv2.resize(reference_image, (100, 100))
# resized_test_image = cv2.resize(test_image, (100, 100))

# # Extract features
# hog_features_reference = extract_hog_features(resized_reference_image)
# hog_features_test = extract_hog_features(resized_test_image)
# lbp_histogram_reference = calculate_lbp_histogram(resized_reference_image)
# lbp_histogram_test = calculate_lbp_histogram(resized_test_image)

# # Concatenate the features
# reference_features = np.concatenate((hog_features_reference, lbp_histogram_reference))
# test_features = np.concatenate((hog_features_test, lbp_histogram_test))

# # Create the training data and labels
# X_train = np.stack((reference_features,))
# y_train = np.array([0,])  # Assuming the reference image is labeled as 0

# # Train a classifier (SVM or Random Forest)
# # classifier = SVC(kernel='linear')
# classifier = RandomForestClassifier(n_estimators=100)
# classifier.fit(X_train, y_train)

# # Make predictions on the test image
# y_pred = classifier.predict(test_features.reshape(1, -1))

# if y_pred[0] == 0:
#     print("The test image matches the reference image.")
# else:
#     print("The test image does not match the reference image.")

# ------------------------- NEW MODEL ---------------------------------------------

import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import euclidean

# Set the path to your dataset directory
dataset_dir = r'E:\Open CV\Alzheimer_s Dataset\train'

# Set the class names
class_names = ['ModerateDemented', 'VeryMildDemented']

# Function to preprocess and extract features from an image
def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    return normalized_image

# Create the training data and labels
X_train = []
y_train = []

# Iterate through each class
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)

    # Iterate through each image in the class directory
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)

        # Load the image
        image = cv2.imread(image_path, 0)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Append the preprocessed image and label to the training data
        X_train.append(processed_image)
        y_train.append(class_name)

# Convert the training data and labels to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Perform one-hot encoding on the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_one_hot = to_categorical(y_train_encoded)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=42)

# Reshape the data to match the input shape of the CNN model
X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
X_val = X_val.reshape(X_val.shape[0], 100, 100, 1)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Load the test image
test_image_path = r'E:\Open CV\Alzheimer_s Dataset\train\ModerateDemented\moderateDem2.jpg'
test_image = cv2.imread(test_image_path, 0)

# Preprocess the test image
processed_test_image = preprocess_image(test_image)

# Reshape the test image to match the input shape of the CNN model
processed_test_image = processed_test_image.reshape(1, 100, 100, 1)

# Make predictions on the test image
predictions = model.predict(processed_test_image)

# Calculate the Euclidean distance between the predicted class vector and each class vector in the training set
distances = [euclidean(predictions[0], class_vector) for class_vector in y_train]

# Get the index of the class with the minimum distance
predicted_class_index = np.argmin(distances)


# Check if the predicted class index is within the range of valid class indices
if predicted_class_index >= len(label_encoder.classes_):
    print("Invalid predicted class index.")
else:
    predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
    print("The test image matches the class:", predicted_class_name)



