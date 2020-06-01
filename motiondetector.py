import os
import sys
import threading
import time
import copy
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


class MotionDetectionParam:
    FRAME_SCALE = 1.0
    # Contour area (WIDTH X HEIGHT)
    OBJECT_SIZE = (100 * 100, 200 * 300)
    NUM_SKIP_FRAME = 0
    ROI = [404, 0, 1070, 680]
    VIDEO_ORG_WIDTH = -1
    VIDEO_ORG_HEIGHT = -1
    # sdaled width
    VIDEO_WIDTH = -1
    # sdaled height
    VIDEO_HEIGHT = -1
    video_source = None


MDP = MotionDetectionParam

input_queue = deque()
output_queue = deque()
display_queue = deque()

event_stop_thread = threading.Event()
event_stop_thread.clear()
event_tracking = threading.Event()
event_tracking.clear()
event_monitor = threading.Event()
event_monitor.clear()


class DisplayData:
    def __init__(self):
        self.input_image = None
        self.model_image = None
        self.fg_image = None
        self.object_image = None
        self.roi_image = None


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
            frame = imutils.resize(frame, int(w * frame_scale))

        disp_data = DisplayData()
        disp_data.input_image = frame
        in_queue.append(disp_data)
        # logging.info(f'queue size : {len(in_queue)}')
    vcap.release()
    logging.info(f'Stopped')


def detect_scene_change(prev_image, curr_image, mask):
    pft = FeatureExtractor(prev_image, MDP.ROI)
    cft = FeatureExtractor(curr_image, MDP.ROI)

    ph = pft.get_histogram(mask)
    ch = cft.get_histogram(mask)

    similarity = cft.compare_histogram(ph, ch)
    scene_changed = similarity >= 0.4

    return scene_changed, similarity


def tracking_thread(curr_image, in_queue, init_bbox, out_queue):
    logging.info(f'Started')

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

    curr_roi_frame = ImageUtil.crop(curr_image, MDP.ROI)

    tracker.init(curr_roi_frame, init_bbox)

    retry_count = 0
    while not event_stop_thread.wait(0.0001):
        try:
            disp_data = in_queue.popleft()
            frame = disp_data.input_image
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count + 1}')
            if retry_count == 5:
                break

            retry_count += 1
            time.sleep(0.5)
            continue

        retry_count = 0

        disp_data.roi_frame = ImageUtil.crop(disp_data.input_image, MDP.ROI)
        (success, box) = tracker.update(disp_data.roi_frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            x = x + MDP.ROI[0]
            y = y + MDP.ROI[1]
            logging.info(f'tracking: {x}, {y}, {w}, {h}')

            disp_data.object_image = copy.copy(disp_data.input_image)
            cv2.rectangle(disp_data.object_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(disp_data.object_image, (MDP.ROI[0], MDP.ROI[1]),
                          (MDP.ROI[2], MDP.ROI[3]),
                          (220, 220, 220), 2)

            out_queue.append(disp_data)
        else:
            logging.info(f'object lost')
            # resume monitor thread
            break

    event_monitor.set()

    logging.info(f'Stopped')


def find_object(fgmask, object_size_threshold):
    lbound = 240
    ubound = 255
    ncon = 0

    # mask = cv2.inRange(fgmask, lbound, ubound)
    _, mask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ncon = len(contours)
    if ncon < 1:
        return [None] * 6

    # method#1 : contour bounding box
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    s = cv2.contourArea(contours[0])
    # logging.info(f'contourArea : {s}')
    if object_size_threshold[1] > s >= object_size_threshold[0]:
        # logging.info(f'contourArea : {s}, min: {object_size_threshold[0]}, max: {object_size_threshold[1]}')
        return x, y, w, h, s, contours[0]

    return [None] * 6


def wait_for_scene_stable(q, w, h, q_window_name=None):
    prev_frame = None
    is_first = True
    stability = 0
    maxlen = 10
    dq = deque([], maxlen)

    # w = Config.VIDEO_WIDTH
    # h = Config.VIDEO_HEIGHT

    nsamples = 100
    widx = np.random.randint(w, size=nsamples)
    hidx = np.random.randint(h, size=nsamples)

    retry_count = 0
    # nskip=1
    while True:
        try:
            disp_data = q.popleft()
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count + 1}')
            if retry_count == 5:
                break
            retry_count += 1
            time.sleep(0.5)
            continue

        if is_first:
            prev_frame = disp_data.input_image
            is_first = False
            continue

        # if nskip % 5 == 0:
        # mse=((frame[hidx, widx[:nsamples], :] - prev_frame[hidx[:nsamples], widx[:nsamples], :]) ** 2).mean(axis=None)
        mse = ((disp_data.input_image[hidx, widx, :] - prev_frame[hidx, widx, :]) ** 2).mean(axis=None)
        dq.append(mse)
        prev_frame = disp_data.input_image
        # logging.info('fetch frame')
        # logging.info(f'dqsize: {len(dq)}/{maxlen}')
        if len(dq) == maxlen:
            v = np.var(dq)
            if v < 40.0:
                stability += 1
            else:
                stability = 0

            if stability > 10:
                logging.info(f'Scene stable : var={v:.2f} stability={stability}')
                return
        # nskip+=1

        if q_window_name:
            display_queue.append(disp_data)


def monitor_thread(in_queue, out_queue, mask):
    default_learning_rate = 0.01

    output_dir = os.path.join(os.path.dirname(MDP.video_source), os.path.splitext(os.path.basename(MDP.video_source))[0])
    logging.info(f'Started')
    logging.info(f'output dir : {output_dir}')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    fgbg.setBackgroundRatio(default_learning_rate)
    # automatically chosen
    learning_rate = -1
    retry_count = 0
    image_count = -1
    prev_image = None
    curr_image = None
    # wait until event is set,
    while event_monitor.wait():
        if event_stop_thread.wait(0.0001):
            break

        try:
            disp_data = in_queue.popleft()
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count + 1}')
            if retry_count == 5:
                break
            retry_count += 1
            time.sleep(0.5)
            continue

        retry_count = 0
        image_count += 1

        disp_data.roi_image = ImageUtil.crop(disp_data.input_image, MDP.ROI)
        disp_data.roi_image = cv2.cvtColor(disp_data.roi_image, cv2.COLOR_BGR2GRAY)
        # not good result
        # roi_frame = cv2.equalizeHist(roi_frame)
        disp_data.roi_image = cv2.GaussianBlur(disp_data.roi_image, None, 3)

        if prev_image is None:
            prev_image = disp_data.roi_image
        else:
            prev_image = curr_image
        curr_image = disp_data.roi_image

        # scene change detection. input color should be preserved
        scene_changed, similarity = detect_scene_change(prev_image, curr_image, mask)
        if scene_changed:
            logging.info(f'scene_changed {image_count:04d}: {similarity}')
            wait_for_scene_stable(in_queue, disp_data.input_image.shape[1], disp_data.input_image.shape[0], None)
            learning_rate = 1
            continue

        if learning_rate == 1:
            logging.info(f'Re-learning scene from now {image_count:04d}')
            # fgbg = None
            # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            fgbg.apply(curr_image, None, 1)
            # automatic learning rate
            # fgbg.setBackgroundRatio()
            learning_rate = default_learning_rate

        fgmask = fgbg.apply(curr_image, None, learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        disp_data.model_image = fgbg.getBackgroundImage()
        disp_data.fg_image = fgmask
        disp_data.object_image = disp_data.input_image

        # TODO: remove debugging code
        cv2.imwrite(f'{output_dir}/{image_count:04d}-model.png', disp_data.model_image)
        cv2.imwrite(f'{output_dir}/{image_count:04d}-foreground.png', disp_data.fg_image)

        # TODO: remove debugging code
        # obj_size=(0, ImageUtil.width(ROI)*ImageUtil.height(ROI))
        obj_size = MDP.OBJECT_SIZE
        x, y, w, h, s, con = find_object(disp_data.fg_image, obj_size)
        if w is not None:
            # logging.info(f'Object detected')

            # METHOD1 : Using tracker
            # th=threading.Thread(None, tracking_thread, "tracking_thread",
            #                     args=(curr_frame, in_queue, (x, y, w, h), out_queue))
            # th.start()
            # event_tracking.set()
            # # pause me
            # event_monitor.clear()

            # METHOD2 : Just print out
            x = x + MDP.ROI[0]
            y = y + MDP.ROI[1]
            # logging.info(f'tracking: {x}, {y}, {w}, {h}')

            cv2.rectangle(disp_data.object_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(disp_data.object_image, (MDP.ROI[0], MDP.ROI[1]), (MDP.ROI[2], MDP.ROI[3]), (220, 220, 220), 2)
            cv2.drawContours(disp_data.object_image, [con + [MDP.ROI[0], MDP.ROI[1]]], 0, (0, 0, 255), thickness=3)
            cv2.putText(disp_data.fg_image, f'contourArea {image_count:04d}: {s}',
                        (0, disp_data.fg_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255))
            cv2.imwrite(f'{output_dir}/{image_count:04d}-object.png', disp_data.object_image)
            logging.info(f'contourArea {image_count:04d}: {s}')

        display_queue.append(disp_data)

    logging.info(f'Stopped {image_count:04d}')

    event_stop_thread.set()


def read_video_params(vsrc):
    vcap = cv2.VideoCapture(vsrc)

    MDP.VIDEO_ORG_WIDTH = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    MDP.VIDEO_ORG_HEIGHT = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    MDP.VIDEO_WIDTH = int(MDP.VIDEO_ORG_WIDTH * MDP.FRAME_SCALE + 0.5)
    MDP.VIDEO_HEIGHT = int(MDP.VIDEO_ORG_HEIGHT * MDP.FRAME_SCALE + 0.5)
    MDP.ROI = (MDP.FRAME_SCALE * np.array(MDP.ROI)).astype(np.int).tolist()
    MDP.OBJECT_SIZE = (MDP.FRAME_SCALE * np.array(MDP.OBJECT_SIZE)).astype(np.int).tolist()

    vcap.release()


def main():

    MDP.video_source = sys.argv[1]
    if not os.path.exists(MDP.video_source):
        logging.info(f'File not found : {MDP.video_source}')
        return

    read_video_params(MDP.video_source)

    ImageUtil.create_image_window("source", MDP.VIDEO_WIDTH, MDP.VIDEO_HEIGHT,
                                  ImageUtil.width(MDP.ROI), ImageUtil.height(MDP.ROI), 0.5)

    mask = ImageLoader.read_image('data/mask1.png')
    mask = imutils.resize(mask, MDP.VIDEO_WIDTH)
    th_capture = threading.Thread(None, capture_thread, "capture_thread",
                                  args=(MDP.video_source, input_queue, MDP.NUM_SKIP_FRAME, MDP.FRAME_SCALE))
    th_monitor = threading.Thread(None, monitor_thread, "monitor_thread",
                                  args=(input_queue, output_queue, mask))

    # start capture
    th_capture.start()

    # start monitor thread
    event_monitor.set()
    th_monitor.start()

    while True:
        if event_stop_thread.wait(0.001):
            break

        if len(display_queue) > 0:
            disp_data = display_queue.popleft()
            # logging.info(f'Updating image')
            cv2.imshow('source', disp_data.input_image)
            cv2.imshow('Model', disp_data.model_image)
            cv2.imshow('Difference', disp_data.fg_image)
            fg_image_color = cv2.cvtColor(disp_data.fg_image, cv2.COLOR_GRAY2BGR)
            model_image_color = cv2.cvtColor(disp_data.model_image, cv2.COLOR_GRAY2BGR)
            disp_data.object_image, nw, nh = ImageUtil.overlay_image(disp_data.object_image, fg_image_color, 0, 0,
                                                                     int(disp_data.object_image.shape[1] * 0.25))
            cv2.rectangle(disp_data.object_image, (0, 0), (nw, nh), (127, 127, 0), 3)
            disp_data.object_image, nw, nh = ImageUtil.overlay_image(disp_data.object_image, model_image_color, 0, nh,
                                                                   int(disp_data.object_image.shape[1] * 0.25))
            cv2.rectangle(disp_data.object_image, (0, nh+3), (nw, 2*nh+3), (0, 127, 127), 3)

            cv2.imshow('Detector', disp_data.object_image)

        if cv2.waitKey(50) == ord('q'):
            break

    event_monitor.set()
    event_stop_thread.set()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main()
