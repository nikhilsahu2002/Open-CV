import pymongo
import os

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



