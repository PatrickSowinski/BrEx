import time
import random
import numpy as np
import cv2

# open video from file
cap = cv2.VideoCapture("../videos/correct_arm_1.mp4")
# open webcam directly
#cap = cv2.VideoCapture(0)

# save most right position of chest and stomach
mostRightChest = -1
mostRightStomach = -1

while(cap.isOpened()):
    ret, frame = cap.read()
    # get dimensions
    imageHeight, imageWidth = frame.shape[:2]
    imageCenterY = int(imageHeight / 2)

    # turn to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayscale', gray)

    # threshold to find contours
    ret, thresh = cv2.threshold(gray, 75, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find largest contours
    areas = [cv2.contourArea(cnt) for cnt in contours]
    index_largest_area = np.argmax(areas)
    largest_contour = contours[index_largest_area]
    #largest_3_indices = np.argsort(areas)[-3:]
    #second_contour = contours[largest_3_indices[1]]
    #third_contour = contours[largest_3_indices[0]]
    #img = cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
    #cv2.imshow('contours', frame)

    # try to find the right part of the contour points and remove the rest
    contour_points = np.squeeze(largest_contour)
    # find right half of points
    xContour = contour_points[:, 0]
    xMedian = np.median(xContour)
    contourRight = np.array([point for point in contour_points if point[0] > xMedian])
    # get top and bottom
    yContour = contourRight[:, 1]
    yTop = np.min(yContour)
    yBottom = np.max(yContour)
    # remove top and bottom 20 pixels
    contourRight = np.array([point for point in contourRight if point[1] > yTop+20 and point[1] < yBottom - 20])

    # separate chest and stomach
    contourChest = np.array([point for point in contourRight if point[1] <= imageCenterY])
    contourStomach = np.array([point for point in contourRight if point[1] > imageCenterY])
    cv2.polylines(frame, [contourChest], False, (255, 0, 0), 2)
    cv2.polylines(frame, [contourStomach], False, (0, 255, 0), 2)

    # get mean chest and stomach position (horizontal)
    xChest = contourChest[:, 0]
    xStomach = contourStomach[:, 0]
    xChestMean = int(np.mean(xChest))
    xStomachMean = int(np.mean(xStomach))
    cv2.line(frame, (xChestMean, 20), (xChestMean, imageCenterY), (0, 0, 255), 2)
    cv2.line(frame, (xStomachMean, imageCenterY), (xStomachMean, imageHeight-20), (0, 0, 255), 2)

    # draw circles for chest and stomach breathing
    if xChestMean > mostRightChest:
        mostRightChest = xChestMean
    if xStomachMean > mostRightStomach:
        mostRightStomach = xStomachMean
    bodyCenter = np.max([mostRightChest, mostRightStomach]) + 100
    # add overlay for transparency of circles
    overlay = frame.copy()
    alpha = 0.4
    cv2.circle(overlay, (bodyCenter, int(imageHeight/4)), bodyCenter-xChestMean, (255, 0, 0), -1)
    cv2.circle(overlay, (bodyCenter, int(3*imageHeight/4)), bodyCenter - xStomachMean, (0, 255, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    cv2.imshow('contours right half', frame)

    """
    # find polygon around largest contour
    epsilon = 0.1*cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    img = cv2.polylines(frame, [polygon], True, (255, 0, 0), 3)
    cv2.imshow('polygon', frame)
    """
    """
    # get corner points
    poly_corners = np.squeeze(polygon)
    xCoords = [point[0] for point in poly_corners]
    yCoords = [point[1] for point in poly_corners]
    # find out which points are bottom/top, left/right
    # (0,0) is top_left, x grows to right, y grows to bottom
    xSorted = np.argsort(xCoords)
    left_points = poly_corners[xSorted[:2]]
    right_points = poly_corners[xSorted[-2:]]
    top_left = left_points[0]
    bottom_left = left_points[1]
    top_right = right_points[0]
    bottom_right = right_points[1]
    """

    # close video with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()