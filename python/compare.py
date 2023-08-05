import pymongo
import numpy as np
import cv2

client = pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
mydb = client['DB']
collection = mydb['tab']

def read_image_as_bytecode(image_path):
    with open(image_path, "rb") as image_file:
        byte_code = image_file.read()
    return byte_code

# Provide the path to the image file
image_path = "yes/Y2.jpg"

# Read the image file as byte code
image_bytecode = read_image_as_bytecode(image_path)

# Store the image data in the database
image_document = {
    "name": "Brain_YES",
    "data": image_bytecode
}
# collection.insert_one(image_document)

# Retrieve the image data from the database
retrieved_document = collection.find_one({"name": "Brain_NO"})
retrieved_bytecode = retrieved_document["data"]

retrieved_document1 = collection.find_one({"name": "Brain_YES"})
retrieved_bytecode1 = retrieved_document1["data"]

# Convert the byte code to a NumPy array
nparr = np.frombuffer(retrieved_bytecode, np.uint8)
nparr1 = np.frombuffer(retrieved_bytecode1, np.uint8)

# Decode the NumPy array as an image using OpenCV
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
image_prv = cv2.resize(image, (400, 300))
image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
image_prv1 = cv2.resize(image1, (400, 300))

# Display the image
cv2.imshow("Image", image_prv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Image", image_prv1)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
    print(f"Similarity: {similarity:.2f}%")

# Compare the images
compare_images(image_prv, image_prv1)

