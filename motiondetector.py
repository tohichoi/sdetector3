import threading
import time

import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image import ImageUtil, VideoUtil, ImageLoader, FeatureExtractor
# from fileutil import *
import logging
from fileutil import create_logging
from mydateutil import DateUtil
from collections import deque

input_queue = deque()
output_queue = deque()

FRAME_SCALE=1.0
# WIDTH X HEIGHT
MIN_OBJECT_SIZE=150*150
NSKIPFRAME=0
ROI = [404, 0, 1006, 680]
VIDEO_ORG_WIDTH = -1
VIDEO_ORG_HEIGHT = -1
VIDEO_WIDTH = -1
VIDEO_HEIGHT = -1
video_source='data/act4.avi'

event_stop_thread=threading.Event()
event_stop_thread.clear()
event_tracking=threading.Event()
event_tracking.clear()
event_monitor=threading.Event()
event_monitor.clear()
event_show_image=threading.Event()
event_show_image.clear()

from threading import BoundedSemaphore
input_image=None
model_image=None
fg_image=None
object_image=None

def capture_thread(video_source, in_queue, nskipframe, frame_scale):
    logging.info(f'Started')
    logging.info(f'Connecting to device {video_source} ...')
    vcap = cv2.VideoCapture(video_source)
    logging.info('Connected.')

    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_skip = 0
    while not event_stop_thread.wait(0.001):
        ret, frame = vcap.read()
        if not ret:
            logging.info('Video decoding error occurred.')
            break
            # vcap = cv2.VideoCapture(Config.video_src)
            # continue

        if frame is None or len(frame) < 1:
            logging.info('no frame.')
            break

        frame_skip += 1
        if nskipframe > 0 and frame_skip % nskipframe != 0:
            continue

        if frame_scale != 1:
            frame=imutils.resize(frame, int(w*frame_scale))

        in_queue.append(frame)
        # logging.info(f'queue size : {len(in_queue)}')
    vcap.release()
    logging.info(f'Stopped')


def detect_scene_change(prev_frame, curr_frame, mask):

    pft=FeatureExtractor(prev_frame, ROI)
    cft=FeatureExtractor(curr_frame, ROI)

    ph=pft.get_histogram(mask)
    ch=cft.get_histogram(mask)

    similarity=cft.compare_histogram(ph, ch)
    scene_changed = similarity >= 0.4

    return scene_changed, similarity


def tracking_thread(curr_frame, in_queue, init_bbox, out_queue):
    logging.info(f'Started')

    global object_image
    global input_image

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    tracker=OPENCV_OBJECT_TRACKERS["csrt"]()

    curr_roi_frame = ImageUtil.crop(curr_frame, ROI)

    tracker.init(curr_roi_frame, init_bbox)

    retry_count=0
    while not event_stop_thread.wait(0.0001):
        try:
            frame=in_queue.popleft()
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count+1}')
            if retry_count == 5:
                break

            retry_count+=1
            time.sleep(0.5)
            continue

        retry_count=0

        roi_frame=ImageUtil.crop(frame, ROI)
        (success, box) = tracker.update(roi_frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            x=x+ROI[0]
            y=y+ROI[1]
            logging.info(f'tracking: {x}, {y}, {w}, {h}')

            input_image=frame

            object_image=frame
            cv2.rectangle(object_image,  (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(object_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (220, 220, 220), 2)

            event_show_image.set()

            out_queue.append(frame)
        else:
            logging.info(f'object lost')
            # resume monitor thread
            break

    event_monitor.set()

    logging.info(f'Stopped')


def find_object(fgmask, object_size_threshold):

    lbound = 240
    ubound = 255

    mask = cv2.inRange(fgmask, lbound, ubound)

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return [None] * 4

    # method#1 : contour bounding box
    # contours = sorted(contours, key=cv2.boundingRect, reverse=True)[:5]
    x, y, w, h = cv2.boundingRect(contours[0])
    if w*h >= object_size_threshold:
        return x, y, w, h

    # method#2 ; contour area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    if cv2.contourArea(contours[0]) and cv2.isContourConvex(contours[0]):
        return x, y, w, h

    return [None] * 4


def monitor_thread(in_queue, out_queue, mask):
    global model_image
    global input_image
    global fg_image
    global object_image

    logging.info(f'Started')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    # automatically chosen
    learning_rate=-1
    retry_count=0
    prev_frame=None
    curr_frame=None
    # wait until event is set,
    while event_monitor.wait():
        if event_stop_thread.wait(0.0001):
            break

        try:
            frame=in_queue.popleft()
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count+1}')
            if retry_count == 5:
                break

            retry_count+=1
            time.sleep(0.5)
            continue

        retry_count=0

        roi_frame=ImageUtil.crop(frame, ROI)
        if prev_frame is None:
            prev_frame=roi_frame
        else:
            prev_frame=curr_frame
        curr_frame=roi_frame

        # scene change detection
        scene_changed, similarity=detect_scene_change(prev_frame, curr_frame, mask)
        if scene_changed:
            logging.info(f'scene_changed: {similarity}')
            learning_rate=1
            continue

        if learning_rate >= 0:
            logging.info(f'Re-learning scene from now')
            fgmask = fgbg.apply(curr_frame, None, 1)
            # automatic learning rate
            learning_rate = -1

        fgmask = fgbg.apply(curr_frame, None, learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # event_show_image.wait()
        model_image=fgbg.getBackgroundImage()
        input_image=frame
        fg_image=fgmask
        object_image=frame
        # event_show_image.set()

        x, y, w, h=find_object(fgmask, MIN_OBJECT_SIZE)
        if w is not None:
            logging.info(f'Object detected')

            # METHOD1 : Using tracker
            # th=threading.Thread(None, tracking_thread, "tracking_thread",
            #                     args=(curr_frame, in_queue, (x, y, w, h), out_queue))
            # th.start()
            # event_tracking.set()
            # # pause me
            # event_monitor.clear()

            # METHOD2 : Just print out
            x=x+ROI[0]
            y=y+ROI[1]
            logging.info(f'tracking: {x}, {y}, {w}, {h}')

            input_image=frame

            object_image=frame
            cv2.rectangle(object_image,  (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(object_image, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (220, 220, 220), 2)

        event_show_image.set()


    logging.info(f'Stopped')

    event_stop_thread.set()


def read_video_params(vsrc):

    global ROI
    global VIDEO_ORG_WIDTH
    global VIDEO_ORG_HEIGHT
    global VIDEO_WIDTH
    global VIDEO_HEIGHT

    vcap=cv2.VideoCapture(vsrc)

    VIDEO_ORG_WIDTH = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDEO_ORG_HEIGHT = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VIDEO_WIDTH = int(VIDEO_ORG_WIDTH * FRAME_SCALE + 0.5)
    VIDEO_HEIGHT = int(VIDEO_ORG_HEIGHT * FRAME_SCALE + 0.5)
    ROI = (FRAME_SCALE * np.array(ROI)).astype(np.int).tolist()

    vcap.release()

def main():

    global input_image
    global model_image
    global fg_image
    global object_image

    read_video_params(video_source)

    ImageUtil.create_image_window("source", VIDEO_WIDTH, VIDEO_HEIGHT,
                                  ImageUtil.width(ROI), ImageUtil.height(ROI), 0.5)

    mask=ImageLoader.read_image('data/mask1.png')
    mask=imutils.resize(mask, VIDEO_WIDTH)
    th_capture=threading.Thread(None, capture_thread, "capture_thread",
                                args=(video_source, input_queue, NSKIPFRAME, FRAME_SCALE))
    th_monitor=threading.Thread(None, monitor_thread, "monitor_thread",
                                args=(input_queue, output_queue, mask))

    # start capture
    th_capture.start()

    # start monitor thread
    event_monitor.set()
    th_monitor.start()

    while True:
        if event_stop_thread.wait(0.001):
            break

        if event_show_image.wait(0.001):
            # logging.info(f'Updating image')
            cv2.imshow('source', input_image)
            cv2.imshow('Model', model_image)
            cv2.imshow('Difference', fg_image)
            cv2.imshow('Detector', object_image)
            event_show_image.clear()
            if cv2.waitKey(100) == ord('q'):
                break

    event_monitor.set()
    event_stop_thread.set()
    event_show_image.clear()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main()