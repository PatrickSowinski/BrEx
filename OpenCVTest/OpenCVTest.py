import time
import random
import numpy as np
import cv2

# open video from file
cap = cv2.VideoCapture("../videos/correct_short_1.mov")
# open webcam directly
#cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 75, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(contours)

    img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('contours', frame)

    cv2.imshow('grayscale', gray)
    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()