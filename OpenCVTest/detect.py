import time
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# open video from file
cap = cv2.VideoCapture(0)
frameCount = 0
lung_area_history = []
belly_area_history = []
#start_time = time.time()
while (cap.isOpened() or (time.time() - start_time) > 10):

    _, img = cap.read()

    if img is None:
        print("No frame available")
        break
    frameCount += 1
    # skip every 2nd frame for performance reasons
    if frameCount % 2 == 0:
        continue


    mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (480, 640)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (int(img.shape[1]/3),10,250,470)

    start_time = time.time()
    # this modifies mask 
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    print("--- GrapCut takes %s seconds ---" % (time.time() - start_time))

    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # adding additional dimension for rgb to the mask, by default it gets 1
    # multiply it with input image to get the segmented image
    img_cut = img*mask2[:,:,np.newaxis]

    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 25, 255, 0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find largest contours
    areas = [cv2.contourArea(cnt) for cnt in contours]
    index_largest_area = np.argmax(areas)
    largest_contour = contours[index_largest_area]
    contour_with_image = cv2.drawContours(img, largest_contour, -1, (255, 0, 255), 3)

    #print(largest_contour.shape)
    #print(largest_contour[0][0])

    topmost = largest_contour[largest_contour[:,:,1].argmin()][0]
    bottommost = largest_contour[largest_contour[:,:,1].argmax()][0]
    x, y = tuple((bottommost-topmost)/2)
    x = int(x)
    y = int(y) + 50
    #print(contour_with_image[0])
    #print(topmost, bottommost)
    
    start = time.time()
    #print(len(contour_with_image[0]))
    lung_area = 0
    belly_area = 0
    for i in range(len(thresh)):
        for j in range(len(thresh[i])):
            if thresh[i][j] == 255:
                if i <= y:
                    lung_area += 1
                    contour_with_image[i][j] = np.array([255, 0, 0])
                else:
                    belly_area += 1
                    contour_with_image[i][j] = np.array([0, 0, 255])
            #else:
                #print(thresh[i][j])
                #contour_with_image[i][j] = np.array([0, 255, 255])
    print("--- Area Calculation takes %s seconds ---" % (time.time() - start))

    #print(lung_area, belly_area ,itr)
    lung_area_history.append(lung_area)
    belly_area_history.append(belly_area)
    #print(lung_area_history)S

    cv2.imshow('contour_with_image', contour_with_image)
    #cv2.imshow('img', img)

    cv2.imshow('thresh', thresh)

    #print("--- %s seconds ---" % (time.time() - start_time))

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

plt.plot(lung_area_history)
plt.plot(belly_area_history)
plt.show()

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