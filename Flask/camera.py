import cv2
import time
import numpy as np
from sklearn.cluster import KMeans

class VideoCamera(object):
    def __init__(self, debug=False):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.contour_found = False
        self.state_array = []
        self.state = ""

        self.chestDiffArray = []
        self.stomachDiffArray = []

        self.frameCount = 0
        # initialize variables for chest and stomach positions
        self.chestMeansArray = []
        self.stomachMeansArray = []
        self.totalChestMean = 0.0
        self.totalStomachMean = 0.0

        self.diff_maxpoint_contour_array = []

        self.debug = debug

        self.color = None

    def __del__(self):
        self.video.release()

    def cut(self):
        print("hello")
        # Color determination only once
        success, frame = self.video.read()

        mask = np.zeros(frame.shape[:2], np.uint8)  # img.shape[:2] = (480, 640)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (int(frame.shape[1] / 3), 10, 250, 470)

        start_time = time.time()
        # this modifies mask
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        print("--- GrapCut takes %s seconds ---" % (time.time() - start_time))

        # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # adding additional dimension for rgb to the mask, by default it gets 1
        # multiply it with input image to get the segmented image
        img_cut = frame * mask2[:, :, np.newaxis]
        # cv2.imshow('img_cut', img_cut)
        rgb_img = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)


        rgb_img = rgb_img.reshape((rgb_img.shape[0] * rgb_img.shape[1], 3))  # represent as row*column,channel number
        clt = KMeans(n_clusters=4)  # cluster number
        clt.fit(rgb_img)
        # hist = find_histogram(clt)
        color = clt.cluster_centers_[1]
        print(color)
        self.color = color

    def get_frame(self):
        success, frame = self.video.read()
        success, debug_frame = self.video.read()


        cor_position = False
        breatheCorrect=False

        if frame is None:
            print("No frame available")
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False,False
        self.frameCount += 1

        # skip every 2nd frame for performance reasons
        #if self.frameCount % 2 == 0:
        #    ret, jpeg = cv2.imencode('.jpg', frame)
        #    return jpeg.tobytes(),False,False
        # get dimensions
        imageHeight, imageWidth = frame.shape[:2]
        imageCenterY = int(imageHeight / 2)

        #colormode = "AUTO"
        colormode = "RED"
        # colormode = "GRAY"
        thresh = frame

        # Alternative 1: find black body
        if colormode == "GRAY":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 75, 255, 0)
            thresh = cv2.bitwise_not(thresh)

        # Alternative 2: find red body
        if colormode == "RED":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

        if colormode == "AUTO":
            H = self.color[0]
            print(H)
            #[count, x] = imhist(Image);
            #H=128 # set manually -delete later
            blurred_frame = cv2.GaussianBlur(frame, (31, 31), 5)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
            # lower mask (0-10)
            lower_color = np.array([int(H*0.8), 0, 50])
            upper_color = np.array([int(H*1.2), 50, 150])
            mask0 = cv2.inRange(hsv, lower_color, upper_color)
            # upper mask (170-180)
            lower_color = np.array([int(H), 0, 50])
            upper_color = np.array([int(H*1.5), 50, 150])
            mask1 = cv2.inRange(hsv, lower_color, upper_color)
            # join my masks
            mask = mask0 + mask1
            thresh = cv2.bitwise_not(mask)
        # cv2.imshow('thresh', thresh)
        '''
        # use morphological opening and closing to cut out noisy parts of mask
        kernelOpen = np.ones((15, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOpen)
        kernelClose = np.ones((15, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelClose)
        thresh = cv2.GaussianBlur(thresh, (15, 15), 5)
        '''
        # find contours from threshold
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if (len(contours) > 0):
            self.contour_found = True
        # skip if no contours
        if len(contours) < 1:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False,False

        # find largest contours (=body)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        index_largest_area = np.argmax(areas)
        largest_contour = contours[index_largest_area]
        extTop = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        self.diff_maxpoint_contour_array.append(extTop[1])
        # cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
        # largest_3_indices = np.argsort(areas)[-3:]
        # second_contour = contours[largest_3_indices[1]]
        # third_contour = contours[largest_3_indices[0]]
        # cv2.imshow('contours', frame)

        # try to find the chest+stomach part of the contour points and remove the rest
        contour_points = np.squeeze(largest_contour)
        # find left half of points
        xContour = contour_points[:, 0]
        xMean = np.mean(xContour)
        xLeft = np.min(xContour)
        contourLeft = np.array([point for point in contour_points if point[0] < (xMean + xLeft) / 2])
        # get top and bottom
        yContour = contourLeft[:, 1]
        yTop = np.min(yContour)
        yBottom = np.max(yContour)
        # remove top and bottom 20 pixels
        nPixels_topbottom = 50
        contourLeft = np.array([point for point in contourLeft if
                                point[1] > yTop + nPixels_topbottom and point[1] < yBottom - nPixels_topbottom])

        # separate chest and stomach
        contourChest = np.array([point for point in contourLeft if point[1] <= imageCenterY])
        if len(contourChest) < 1:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False,False
        contourStomach = np.array([point for point in contourLeft if point[1] > imageCenterY])
        if len(contourStomach) < 1:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False,False
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
        self.chestMeansArray.append(xChestMean)
        self.stomachMeansArray.append(xStomachMean)
        # update mean of means with soft step (or just mean for first 100 frames)
        if self.frameCount < 100:
            self.totalChestMean = np.mean(self.chestMeansArray)
            self.totalStomachMean = np.mean(self.stomachMeansArray)
        else:
            updateFactor = 0.005
            self.totalChestMean = updateFactor * xChestMean + (1-updateFactor) * self.totalChestMean
            self.totalStomachMean = updateFactor * xStomachMean + (1 - updateFactor) * self.totalStomachMean
        cv2.line(frame, (int(self.totalChestMean), 20), (int(self.totalChestMean), imageCenterY), (0, 0, 255), 1)
        cv2.line(frame, (int(self.totalStomachMean), imageCenterY), (int(self.totalStomachMean), imageHeight - 20), (0, 0, 255), 1)
        # draw circles for chest and stomach breathing
        chestCenter = int(self.totalChestMean + 80)
        chestRadius = chestCenter - xChestMean
        stomachCenter = int(self.totalStomachMean + 80)
        stomachRadius = stomachCenter - xStomachMean
        if chestRadius<0 or stomachRadius<0:
            # reset total means
            self.totalChestMean = xChestMean
            self.totalStomachMean = xStomachMean
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False,False
        # add overlay for transparency of circles
        overlay = frame.copy()
        alpha = 0.4
        cv2.circle(overlay, (chestCenter, int(3*imageHeight/8)-10), chestRadius, (255, 0, 0), -1)
        cv2.circle(overlay, (stomachCenter, int(3*imageHeight/4)), stomachRadius, (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        # save diff of chest and stomach (= total mean - current)
        chestDiff = self.totalChestMean - xChestMean #if + inhale if - exhale
        self.chestDiffArray.append(chestDiff)
        stomachDiff = self.totalStomachMean - xStomachMean #if + inhale if - exhale
        self.stomachDiffArray.append(stomachDiff)

        # check if diff is unusually large
        if max(abs(chestDiff), abs(stomachDiff)) > imageHeight/6:
            # reset total means
            self.totalChestMean = xChestMean
            self.totalStomachMean = xStomachMean

        # check which part has a larger diff
        # (average over last 20 frames)
        nFrames_diff = 80
        if len(self.chestDiffArray) >= nFrames_diff:

            chestDiffAvg_mean_last3 = int(np.mean(self.chestDiffArray[-10:]))
            chestDiffAvg_mean_first3 = int(np.mean(self.chestDiffArray[-20:-10]))
            stomachDiffAvg_mean_last3 = int(np.mean(self.stomachDiffArray[-10:]))
            stomachDiffAvg_mean_first3 = int(np.mean(self.stomachDiffArray[-20:-10]))

            chest_expansion_rate = int(chestDiffAvg_mean_first3 - chestDiffAvg_mean_last3)
            stomach_expansion_rate = int(stomachDiffAvg_mean_first3 - stomachDiffAvg_mean_last3)


            diff_maxpoint_contour_max = int(np.max(self.diff_maxpoint_contour_array[-10:]))
            diff_maxpoint_contour_min = int(np.min(self.diff_maxpoint_contour_array[-20:-10]))


            '''
            #print ("Difference between chest and stomach expansion" + str(chest_expansion_rate - stomach_expansion_rate))

            print(int(self.diff_maxpoint_contour_array_mean_first3 - self.diff_maxpoint_contour_array_mean_last3))
            if (int(self.diff_maxpoint_contour_array_mean_first3 - self.diff_maxpoint_contour_array_mean_last3)) > 0:
                self.state = "lung_inhale"
                print("lung_inhale")
            elif (int(self.diff_maxpoint_contour_array_mean_first3 - self.diff_maxpoint_contour_array_mean_last3)) < 0:
                self.state = "lung_exhale"
                print("lung_exhale")
            else:
                if (stomach_expansion_rate > 0):
                    self.state = "belly_inhale"
                    print("belly_inhale")
                elif (stomach_expansion_rate < 0):
                    self.state = "belly_exhale"
                    print("belly_exhale")
                else:
                    self.state = "holding"
                    print("holding")
            '''

            if (chest_expansion_rate > 0) and (stomach_expansion_rate > 0):
                if (chest_expansion_rate - stomach_expansion_rate) > 0:
                    self.state_array.append("lung_exhale")
                    #print("lung_exhale")
                else:
                    self.state_array.append("belly_exhale")
                    #print("belly_exhale")
            elif (chest_expansion_rate < 0) and (stomach_expansion_rate < 0):
                if (chest_expansion_rate - stomach_expansion_rate) < 0:
                    self.state_array.append("lung_inhale")
                    #print("lung_inhale")
                else:
                    self.state_array.append("belly_inhale")
                    #print("belly_inhale")
                '''
            elif (chest_expansion_rate > 0) or (stomach_expansion_rate < 0):
                    self.state_array.append("lung_exhale")
            elif (chest_expansion_rate < 0) or (stomach_expansion_rate > 0):
                    self.state_array.append("lung_inhale")
                '''
            else:
                self.state_array.append("holding")
                #print("holding")

            if len(self.state_array) > 80:
                self.state = max(set(self.state_array[-5:]), key = self.state_array.count)
                print (self.state)


        # close video with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(),False, False
        cor_position=True
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        if self.debug:
            ret, jpeg = cv2.imencode('.jpg', debug_frame)
            return jpeg.tobytes(),False,False

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), cor_position, breatheCorrect
        #cap.release()
        #cv2.destroyAllWindows()
        # print("Total frames:", frameCount)
