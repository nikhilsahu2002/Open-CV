import cv2

def compare_images(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Check if the images are loaded successfully
    if image1 is None or image2 is None:
        print("Failed to load the images.")
        return

    # Resize the images to a common size (optional)
    image1 = cv2.resize(image1, (400, 300))
    image2 = cv2.resize(image2, (400, 300))

    

    # Compute the absolute difference between the images
    diff = cv2.absdiff(image1, image2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, the images are identical
    if len(contours) == 0:
        print("The images are identical.")
    else:
        print("The images are different.")

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.waitKey(0) 

# Provide the paths to the images you want to compare
image1_path = "1.jpg"
image2_path = "1.jpg"

# Call the function to compare the images
compare_images(image1_path, image2_path)
