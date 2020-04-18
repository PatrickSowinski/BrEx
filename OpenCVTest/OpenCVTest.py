import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# open video from file
cap = cv2.VideoCapture("../videos/correct_short_1.mp4")
# open webcam directly
#cap = cv2.VideoCapture(0)

# save most right position of chest and stomach
mostRightChest = -1
mostRightStomach = -1

frameCount = 0
# initialize variables for chest and stomach positions
chestMeansArray = []
stomachMeansArray = []
totalChestMean = 0.0
totalStomachMean = 0.0

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print("No frame available")
        break
    frameCount += 1
    # skip every 2nd frame for performance reasons
    if frameCount % 2 == 0:
        continue
    # get dimensions
    imageHeight, imageWidth = frame.shape[:2]
    imageCenterY = int(imageHeight / 2)

    # initialize chest and stomach positions at 3/4 to the right (first guess)
    if totalChestMean == 0:
        totalChestMean = 3*imageWidth/4
    if totalStomachMean == 0:
        totalStomachMean = 3*imageWidth/4

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

    # save chest and stomach means
    chestMeansArray.append(xChestMean)
    stomachMeansArray.append(xStomachMean)

    # update mean of means with soft step (or just mean for first 100 frames)
    if frameCount < 100:
        totalChestMean = int(np.mean(chestMeansArray))
        totalStomachMean = int(np.mean(stomachMeansArray))
    else:
        updateFactor = 0.02
        totalChestMean = updateFactor * xChestMean + (1-updateFactor) * totalChestMean
        totalStomachMean = updateFactor * xStomachMean + (1 - updateFactor) * totalStomachMean

    cv2.line(frame, (int(totalChestMean), 20), (int(totalChestMean), imageCenterY), (0, 0, 255), 1)
    cv2.line(frame, (int(totalStomachMean), imageCenterY), (int(totalStomachMean), imageHeight - 20), (0, 0, 255), 1)

    # draw circles for chest and stomach breathing
    chestCenter = int(totalChestMean + 80)
    chestRadius = chestCenter - xChestMean
    stomachCenter = int(totalStomachMean + 80)
    stomachRadius = stomachCenter - xStomachMean
    if chestRadius<0 or stomachRadius<0:
        continue
    # add overlay for transparency of circles
    overlay = frame.copy()
    alpha = 0.4
    cv2.circle(overlay, (chestCenter, int(imageHeight/4)), chestRadius, (255, 0, 0), -1)
    cv2.circle(overlay, (stomachCenter, int(3*imageHeight/4)), stomachRadius, (0, 255, 0), -1)
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
print("Total frames:",frameCount)

# plot means of chest and stomach position
# create an array with frame numbers, since we skipped every 2nd one
frameArray = 2*np.array(range(len(chestMeansArray)))
chestPlot = plt.plot(frameArray, chestMeansArray, label="Chest")
stomachPlot = plt.plot(frameArray, stomachMeansArray, label="Stomach")
_, _, ymin, ymax = plt.axis()
plt.ylim([ymin, min(ymax, int(3*imageWidth/4))])
plt.legend()
plt.ylabel("Horizontal position [pixels]")
plt.xlabel("Frame")
plt.show()