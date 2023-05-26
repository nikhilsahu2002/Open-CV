import pymongo
import numpy as np
import cv2

client=pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
my=client['DB']
info=my.tab

def read_image_as_bytecode(image_path):
    with open(image_path, "rb") as image_file:
        byte_code = image_file.read()
    return byte_code

# Provide the path to the image file
image_path = "1.jpg"

# Read the image file as byte code
image_bytecode = read_image_as_bytecode(image_path)

# Store the image data in the database
image_document = {
    "name": "Image 1",
    "data": image_bytecode
}
# info.insert_one(image_document)

# Retrieve the image data from the database
retrieved_document = info.find_one({"name": "Image 1"})
retrieved_bytecode = retrieved_document["data"]

# Convert the byte code to a NumPy array
nparr = np.frombuffer(retrieved_bytecode, np.uint8)

# Decode the NumPy array as an image using OpenCV
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
image_prv = cv2.resize(image,(400,300))


# Display the image
cv2.imshow("Image", image_prv)
cv2.waitKey(0)
cv2.destroyAllWindows()

