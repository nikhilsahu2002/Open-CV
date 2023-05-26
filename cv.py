import cv2
import numpy as np

def compare_images(image1, image2):
    # Load the images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
     # Resize the images to the same dimensions
    img1 = cv2.resize(img1, (800, 600))
    img2 = cv2.resize(img2, (800, 600))

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around the contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the images
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the percentage of similarity
    similarity = (1 - (np.count_nonzero(diff) / diff.size)) * 100
    print(f"Similarity: {similarity:.2f}%")

# Provide the paths to the images you want to compare
image_path1 = "imag1.png"
image_path2 = "imag2.png"

# Compare the images
compare_images(image_path1, image_path2)
