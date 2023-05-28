import pymongo
import os
import numpy as np
import cv2

client = pymongo.MongoClient('mongodb://localhost:27017/?directConnection=true')
mydb = client['DB']
collection = mydb['tab']

def read_image_as_bytecode(image_path):
    with open(image_path, "rb") as image_file:
        byte_code = image_file.read()
    return byte_code

# Path to the folder containing the images
folder_path = "yes"

# Iterate over each image file in the folder
for file_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, file_name)

    # Check if the file is an image
    if os.path.isfile(image_path) and file_name.lower().endswith(('.jpg','JPG', '.jpeg', '.png')):
        image_bytecode = read_image_as_bytecode(image_path)
        image_document = {
            "name": file_name,
            "data": image_bytecode
        }
        # collection.insert_one(image_document)

# print("Image data stored in MongoDB successfully.")

retrive_document = collection.find_one({"name": "Y54.jpg"})

# Check if the document exists
if retrive_document is not None:
    # Retrieve the image data from the document
    retrive_bytercod = retrive_document.get("data")

    if retrive_bytercod is not None:
        # Convert the byte code to a NumPy array
        nparr = np.frombuffer(retrive_bytercod, np.uint8)

        # Decode the NumPy array as an image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_prv = cv2.resize(image, (400, 300))

        # Display the image
        cv2.imshow("Image", image_prv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image document does not contain 'data' field.")
else:
    print("Document with name 'Y54' not found.")

