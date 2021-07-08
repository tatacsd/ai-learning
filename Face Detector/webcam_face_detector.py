import cv2
# https://github.com/opencv/opencv/tree/master/data/haarcascades
from random import randrange

# Load the pre-trained data on face from opencv (haar cascade algorithm)
# Use cv2 import and the function Cascade Classifier (detector)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Load webcam to be detected
webcam = cv2.VideoCapture(0)  # zero for default webcam


# Iterate over all frames of video
# Iterate over all frames of video
while True:
    # Read the frame
    successful_frame_read, frame = webcam.read()  # tuple boolean and the frame

    # Convert to greyscale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Algorith to detect the faces coordinates in any scale
    face_coordinates = trained_face_data.detectMultiScale(grey_frame)

    # For multiperson
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display Image with dectector
    cv2.imshow('Face Detector program', frame)

    # Wait for user to press any key
    key = cv2.waitKey(1)

    # If "q" is pressed Stop
    if key == 81 or key == 113:
        break

    # realease the webcam
    webcam.release()

print("Code Completed")
