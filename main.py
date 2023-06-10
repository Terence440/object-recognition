import cv2
import numpy as np
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("multi.jpeg")

# OpenCV opens images as BGR,
# but we want it as RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply image transformations
# (e.g., rotation, resizing, scaling etc.)
# to improve object detection accuracy

# Apply rotation transformation
# rotation_angle = 30  # Specify the rotation angle (in degrees)
# rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotation_angle, 1)
# img_gray_rotated = cv2.warpAffine(img_gray, rotation_matrix, (img.shape[1], img.shape[0]))

# Use a cascade classifier for object detection
stop_data = cv2.CascadeClassifier('stop_data.xml')

# Adjust the scaleFactor, minNeighbors, and minSize parameters
found = stop_data.detectMultiScale(img_gray, scaleFactor=1.05,
                                   minNeighbors=1, minSize=(30, 30))

print("Object found:")
print(found)

# Don't do anything if there's no sign
amount_found = len(found)

if amount_found != 0:
    # There may be more than one sign in the image
    for (x, y, width, height) in found:
        # Draw a green rectangle around every recognized sign
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

# Create the environment of the picture and show it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()
