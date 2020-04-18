import time
import random
import numpy as np
import cv2


#dealing with pictures
thresh = 75
im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,thresh,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow('imag', im)
#cv2.waitKey()

########################################################################


# open video from file
#cap = cv2.VideoCapture("../Hack2020_breathing_mp4.mp4")
##open webcam directly
##cap = cv2.VideoCapture(0)

#while (cap.isOpened()):
   # ret, frame = cap.read()


#########################################################################
# turn to grayscale
frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
dim= np.array(frame.shape)
max_row=dim[0]-1
max_col=dim[1]-1

frame[0:10,:]=255
frame[max_row,:]=255
frame[:,0]=255
frame[:,max_col]=255


cv2.imshow('grayscale_pad', frame)
cv2.waitKey()

# threshold to find contours (old approach)

#print(contours)
#blurred_frame=cv2.GaussianBlur(frame, (5,5), 0)
lower_black = np.array([85])
higher_black = np.array([255])
mask = cv2.inRange(frame, lower_black, higher_black)
contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Mask', mask)

output = cv2.bitwise_and(frame, frame, mask=mask)
contours3 = cv2.drawContours(im, contours2, -1, 255, 3)

cv2.imshow('Mask3', contours3)
cv2.waitKey()

#calculating the biggest contour
# sorting the list
print("contours2",np.array(contours2).shape)
c=contours2.sort()

# printing the second last element
print("Second largest element is:", c[-2])

'''
areas = [cv2.contourArea(c) for c in contours2]
print(areas)
max_index = np.argmax(areas)
cnt=contours[max_index]
print(cnt)
'''



'''
# find largest contours
areas = [cv2.contourArea(cnt) for cnt in contours]
index_largest_area = np.argmax(areas)
largest_contour = contours[index_largest_area]
        # largest_3_indices = np.argsort(areas)[-3:]
        # second_contour = contours[largest_3_indices[1]]
        # third_contour = contours[largest_3_indices[0]]
        # img = cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
        # cv2.imshow('contours', frame)

# find polygon around largest contour
epsilon = 0.1 * cv2.arcLength(largest_contour, True)
polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
img = cv2.polylines(gray, [polygon], True, (255, 0, 0), 3)
cv2.imshow('polygon', gray)
cv2.waitKey()
# get corner points
poly_squeezed = np.squeeze(polygon)
xCoords = [point[0] for point in poly_squeezed]
yCoords = [point[1] for point in poly_squeezed]
# print(xCoords)
# print(yCoords)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

#cap.release()
#cv2.destroyAllWindows()

'''
