import cv2
# https://github.com/opencv/opencv/tree/master/data/haarcascades
from random import randrange

# Load the pre-trained data on face from opencv (haar cascade algorithm)
# Use cv2 import and the function Cascade Classifier (detector)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Load an image to be detected
img = cv2.imread('group_picture.jpg')
# img = cv2.imread('group_picture2.jpg')

# Convert to greyscale
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Algorith to detect the faces coordinates in any scale
face_coordinates = trained_face_data.detectMultiScale(grey_img)

# Draw rectangle in the given coordinates
# for a single picture
# (x, y, w, h) = face_coordinates[0]
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# For multiperson
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 2)


# Display Image with dectector
cv2.imshow('Face Detector program', img)

# Wait for user to press any key
cv2.waitKey()

print("Code Completed")
