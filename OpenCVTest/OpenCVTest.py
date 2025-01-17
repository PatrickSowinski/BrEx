import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# open video from file
#cap = cv2.VideoCapture("../videos/correct_arm_1.mp4")
# open webcam directly
cap = cv2.VideoCapture(0)

frameCount = 0
# initialize variables for chest and stomach positions
chestMeansArray = []
stomachMeansArray = []
chestDiffArray = []
stomachDiffArray = []
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

    colormode = "RED"
    #colormode = "GRAY"
    thresh = frame

    # Alternative 1: find black body
    if colormode == "GRAY":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 75, 255, 0)
        thresh = cv2.bitwise_not(thresh)

    # Alternative 2: find red body
    if colormode == "RED":
        blurred_frame = cv2.GaussianBlur(frame, (31, 31), 5)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        # lower mask (0-10)
        lower_red = np.array([0, 100, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)
        # upper mask (170-180)
        lower_red = np.array([170, 100, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # join my masks
        mask = mask0 + mask1
        thresh = mask

    # find contours from threshold
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("Threshold", thresh)

    # skip if no contours
    if len(contours) < 1:
        continue

    # find largest contours (=body)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    index_largest_area = np.argmax(areas)
    largest_contour = contours[index_largest_area]
    #cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
    #largest_3_indices = np.argsort(areas)[-3:]
    #second_contour = contours[largest_3_indices[1]]
    #third_contour = contours[largest_3_indices[0]]
    #cv2.imshow('contours', frame)

    # try to find the chest+stomach part of the contour points and remove the rest
    contour_points = np.squeeze(largest_contour)
    # find left half of points
    xContour = contour_points[:, 0]
    xMean = np.mean(xContour)
    xLeft = np.min(xContour)
    contourLeft = np.array([point for point in contour_points if point[0] < (xMean+xLeft)/2])
    # get top and bottom
    yContour = contourLeft[:, 1]
    yTop = np.min(yContour)
    yBottom = np.max(yContour)
    # remove top and bottom 20 pixels
    nPixels_topbottom = 50
    contourLeft = np.array([point for point in contourLeft if point[1] > yTop+nPixels_topbottom and point[1] < yBottom - nPixels_topbottom])

    # separate chest and stomach
    contourChest = np.array([point for point in contourLeft if point[1] <= imageCenterY])
    contourStomach = np.array([point for point in contourLeft if point[1] > imageCenterY])
    cv2.polylines(frame, [contourChest], False, (255, 0, 0), 2)
    cv2.polylines(frame, [contourStomach], False, (0, 255, 0), 2)
    """
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
        totalChestMean = np.mean(chestMeansArray)
        totalStomachMean = np.mean(stomachMeansArray)
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

    # save diff of chest and stomach (= total mean - current)
    chestDiff = totalChestMean - xChestMean
    chestDiffArray.append(chestDiff)
    stomachDiff = totalStomachMean - xStomachMean
    stomachDiffArray.append(stomachDiff)

    # check which part has a larger diff
    # (average over last 20 frames)
    nFrames_diff = 20
    if len(chestDiffArray) >= nFrames_diff:
        chestDiffAvg = np.mean(chestDiffArray[-nFrames_diff:])
        stomachDiffAvg = np.mean(stomachDiffArray[-nFrames_diff:])
        breatheCorrect = (abs(stomachDiffAvg) > abs(chestDiffAvg))
        if breatheCorrect:
            cv2.circle(frame, (50, 50), 10, (0, 255, 0), -1)
    """
    cv2.imshow('contours', frame)

    # close video with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Total frames:",frameCount)

# PLOTS
# create an array with frame numbers, since we skipped every 2nd one
frameArray = 2*np.array(range(len(chestMeansArray)))
# plot means of chest and stomach position
chestPlot = plt.plot(frameArray, chestMeansArray, label="Chest")
stomachPlot = plt.plot(frameArray, stomachMeansArray, label="Stomach")
# clip ymax, because of stupid peaks
_, _, ymin, ymax = plt.axis()
plt.ylim([ymin, min(ymax, int(3*imageWidth/4))])
plt.legend()
plt.title("Current horizontal position")
plt.ylabel("Horizontal position [pixels]")
plt.xlabel("Frame")
plt.show()
# plot means of chest and stomach position
frameArray = 2*np.array(range(len(chestDiffArray)))
chestPlot = plt.plot(frameArray, chestDiffArray, label="Chest")
stomachPlot = plt.plot(frameArray, stomachDiffArray, label="Stomach")
plt.legend()
plt.title("Difference between current position and mean position")
plt.ylabel("Horizontal position [pixels]")
plt.xlabel("Frame")
plt.show()