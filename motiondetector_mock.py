import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image import ImageUtil, VideoUtil
# from fileutil import *
import logging
from fileutil import create_logging
from mydateutil import DateUtil
from collections import deque



frame_q=deque()

create_logging()

roi = [404, 0, 1006, 680]
source='../sdetector/data/act5.mp4'
# source='rtsp://admin:!@90ASkl@192.168.2.115/11'
#
cap = cv2.VideoCapture(source)


logging.info(source)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=20, detectShadows = False)

# fgbg.setHistory(240)

print(DateUtil.get_current_timestring())

w0, h0=VideoUtil.get_size(source)
x1, y1, w1, h1=ImageUtil.coord(roi)
# ImageUtil.create_image_window(source, w0, h0, w1, h1, 0.5)
connectivity = 4
min_thresh=800
max_thresh=10000
idx=-1


fig, ax=plt.subplots(1, 3)
ax[0].imshow(np.zeros((h0, w0, 1)))

prev_frame=None
curr_frame=None
# plt.ion()
# plt.show()
while(1):
    ret, frame = cap.read()

    idx+=1
    if idx % 100 == 0:
        print(idx)

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imutils.resize(w)

    gray=ImageUtil.crop(gray, roi)
    gray=cv2.medianBlur(gray, 5)
    gray=cv2.equalizeHist(gray)

    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # output=cv2.connectedComponentsWithStats(
    #     fgmask, connectivity, cv2.CV_32S)
    #
    # for i in range(output[0]):
    #     if output[2][i][4] >= min_thresh and output[2][i][4] <= max_thresh:
    #         cv2.rectangle(frame, (output[2][i][0], output[2][i][1]), (
    #             output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)

    ax[0].set_data(orgframe)

    # plt.subplot(1, 3, 1)
    # plt.imshow(orgframe)
    # plt.subplot(1, 3, 2)
    # plt.imshow(gray)
    # plt.subplot(1, 3, 3)
    # plt.imshow(fgmask)
    # plt.pause(0.01)
    # cv2.imshow(source, orgframe)
    # cv2.imshow('Model', gray)
    # cv2.imshow('Difference',fgmask)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

    plt.show(block = False)
    # plt.pause(.001)


cap.release()
cv2.destroyAllWindows()