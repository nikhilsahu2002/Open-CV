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
collection.insert_one(image_document)