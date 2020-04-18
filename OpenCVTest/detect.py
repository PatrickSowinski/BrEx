import time
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt

# open video from file
cap = cv2.VideoCapture(0)
while True:

    _, img = cap.read()
    mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (480, 640)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (160,0,320,480)

    # this modifies mask 
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # adding additional dimension for rgb to the mask, by default it gets 1
    # multiply it with input image to get the segmented image
    img_cut = img*mask2[:,:,np.newaxis]

    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, 0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find largest contours
    areas = [cv2.contourArea(cnt) for cnt in contours]
    index_largest_area = np.argmax(areas)
    largest_contour = contours[index_largest_area]
    contour_with_image = cv2.drawContours(img, largest_contour, -1, (255, 0, 255), 3)

    print(largest_contour.shape)
    print(largest_contour[0][0])

    topmost = largest_contour[largest_contour[:,:,1].argmin()][0]
    bottommost = largest_contour[largest_contour[:,:,1].argmax()][0]
    x, y = tuple((bottommost-topmost)/2)
    y = int(y)
    print(contour_with_image[0])
    #print(topmost, bottommost)
    
    lung_area
    belly_area


    cv2.imshow('contour_with_image', contour_with_image)
    #cv2.imshow('img', img)

    cv2.imshow('thresh', thresh)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

#    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)

#    (B, G, R) = cv2.split(img_cut)

#    maxB = np.amax(B)
#    maxG = np.amax(G)
#    maxR = np.amax(R)
#    minB = np.amin(B)
#    minG = np.amin(G)
#    minR = np.amin(R)

#    higher_black = np.array([maxB, maxG, maxR])
#    lower_black = np.array([minB, minG, minR])
#    mask = cv2.inRange(img, lower_black, higher_black)