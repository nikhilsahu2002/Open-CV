import pymongo
import numpy as np
import cv2
import math

client = pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
mydb = client['DB']
collection = mydb['tab']

def read_image_as_bytecode(image_path):
    with open(image_path, "rb") as image_file:
        byte_code = image_file.read()
    return byte_code

def compare_images(image1, image2):
    # Resize the images to the same dimensions
    image1 = cv2.resize(image1, (800, 600))
    image2 = cv2.resize(image2, (800, 600))

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of similarity
    similarity = (1 - (np.count_nonzero(diff) / diff.size)) * 100
    return similarity

# Retrieve all image documents from the collection
image_documents = collection.find()

total_similarity = 0
image_count = 0

max_similarity = 0
max_similarity_image = None

# Iterate over the image documents and compare with new data
for image_doc in image_documents:
    # Check if "data" key exists in the document
    if "data" in image_doc:
        # Retrieve the image data from the document
        retrieved_bytecode = image_doc["data"]

        # Convert the byte code to a NumPy array
        nparr = np.frombuffer(retrieved_bytecode, np.uint8)

        # Decode the NumPy array as an image using OpenCV
        retrieved_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform the image comparison with the new data (replace "new_image_path" with the path to the new image)
        new_image_path = "k.jpg"
        new_image_bytecode = read_image_as_bytecode(new_image_path)
        nparr_new = np.frombuffer(new_image_bytecode, np.uint8)
        new_image = cv2.imdecode(nparr_new, cv2.IMREAD_COLOR)
        similarity = compare_images(retrieved_image, new_image)

        total_similarity += similarity
        image_count += 1

        if similarity > max_similarity:
            max_similarity = similarity
            max_similarity_image = image_doc    


        print(f"Similarity with {image_doc['name']}: {similarity:.2f}%")
    else:
        print("Image document does not contain 'data' field.")

# if image_count > 0:
#     avg_similarity = total_similarity / image_count
#     print(f"Average Similarity: {avg_similarity:.2f}%")
# else:
#     print("No images found in the database.")

if max_similarity_image is not None:
    print(f"Max Similarity: {max_similarity:.2f}% with image {max_similarity_image['name']}")
else:
    print("No images found in the database.")