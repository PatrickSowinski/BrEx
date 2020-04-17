import time
import random
import numpy as np
import cv2

# open video from file
cap = cv2.VideoCapture("../videos/correct_arm_1.mp4")
# open webcam directly
#cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 75, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    index_largest_area = np.argmax(areas)
    largest_contour = contours[index_largest_area]
    #largest_3_indices = np.argsort(areas)[-3:]
    #second_contour = contours[largest_3_indices[1]]
    #third_contour = contours[largest_3_indices[0]]

    img = cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
    cv2.imshow('contours', frame)

    #polygon = cv2.

    #cv2.imshow('grayscale', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()