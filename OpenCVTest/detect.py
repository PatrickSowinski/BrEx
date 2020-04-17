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

    cv2.imshow('img', img_cut)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()