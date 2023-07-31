import tensorflow as tf
import matplotlib.pyplot as plt
import os

def apply_data_augmentation(image_path, output_directory, num_augmented_images=100):
    # Load the image from the file path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the Keras model with RandomZoom layer
    data_augmentation_model = tf.keras.Sequential([
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ])

    # Apply data augmentation and save the images
    for i in range(num_augmented_images):
        # Apply the random zoom using the Keras model
        augmented_image = data_augmentation_model(tf.expand_dims(image, 0))
        augmented_image = augmented_image[0]

        # Randomly flip the image horizontally and/or vertically
        augmented_image = tf.image.random_flip_left_right(augmented_image)
        augmented_image = tf.image.random_flip_up_down(augmented_image)

        # Randomly adjust brightness
        augmented_image = tf.image.random_brightness(augmented_image, max_delta=0.2)

        # Save the augmented images
        filename = f"augmented_image_{i}.png"
        image_path = os.path.join(output_directory, filename)

        # Convert the image back to uint8 and save it
        image_uint8 = tf.image.convert_image_dtype(augmented_image, tf.uint8)
        encoded_image = tf.io.encode_png(image_uint8)
        tf.io.write_file(image_path, encoded_image)

if __name__ == "__main__":
    # Specify the input image file path
    # input_image_path = r"E:\Open CV\Screenshot (314).png"  # Replace with the actual image file path
    input_image_path = r"E:\Open CV\Screenshot (318).png"  # Replace with the actual image file path

    # Specify the output directory to save augmented images
    output_directory = "autisum_test"

    # Number of augmented images to create
    num_augmented_images = 100

    # Apply data augmentation and save the augmented images
    apply_data_augmentation(input_image_path, output_directory, num_augmented_images)
