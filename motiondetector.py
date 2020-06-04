import re
import os
import sys
import threading
import time
import copy
from datetime import datetime
from queue import Queue
import json
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image import ImageUtil, VideoUtil, ImageLoader, FeatureExtractor
# from fileutil import *
import logging
from fileutil import create_logging, VideoFileWritingThread, FileUtil
from mydateutil import DateUtil
from collections import deque


class MotionDetectionParam:
    FRAME_SCALE = 1.0
    # Contour area (WIDTH X HEIGHT)
    OBJECT_SIZE = (100 * 100, 200 * 300)
    NUM_SKIP_FRAME = 0
    ROI = [404, 0, 1070, 680]
    SCENE_CHANGE_THRESHOLD = 0.4
    MAX_OBJECT_SLICE = 200
    MOVING_WINDOW_SIZE = 20
    VIDEO_ORG_WIDTH = -1
    VIDEO_ORG_HEIGHT = -1
    # scaled width
    VIDEO_WIDTH = -1
    # scaled height
    VIDEO_HEIGHT = -1
    video_source = None
    output_dir = None
    DEBUG = True
    DEBUG_FRAME_TO_FILE = False
    FPS = None


MDP = MotionDetectionParam

input_queue = deque()
display_queue = deque()
object_list = deque()
file_queue = Queue()

event_stop_thread = threading.Event()
event_stop_thread.clear()
event_tracking = threading.Event()
event_tracking.clear()
event_monitor = threading.Event()
event_monitor.clear()
event_object_writing = threading.Event()
event_object_writing.clear()
event_capture_ready = threading.Event()
event_capture_ready.clear()


# millisec
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


# from opencv sample
class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v


class DisplayData:
    def __init__(self, index=0, timestamp=0):
        self.index = index
        self.time = timestamp if timestamp > 0 else datetime.now()
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

    logging.info(f'Estimating FPS')
    hw_fps, fps = VideoUtil.estimate_fps(vcap)
    logging.info(f'FPS : {hw_fps:.1f}(device), {fps:.1f}(estimated)')
    MDP.FPS = min(hw_fps, fps)

    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    event_capture_ready.set()

    frame_index = 0
    while not event_stop_thread.wait(0.001):
        ret, frame = vcap.read()
        if not ret:
            logging.info('Video decoding error occurred.')
            break

        if frame is None or len(frame) < 1:
            logging.info('no frame.')
            break

        frame_index += 1
        if nskipframe > 0 and frame_index % nskipframe != 0:
            continue

        if frame_scale != 1:
            frame = imutils.resize(frame, int(w * frame_scale))

        display_data = DisplayData(index=frame_index - 1)
        display_data.input_image = frame
        in_queue.append(display_data)

    vcap.release()
    logging.info(f'Stopped')


def detect_scene_change(prev_image, curr_image, mask):
    pft = FeatureExtractor(prev_image, MDP.ROI)
    cft = FeatureExtractor(curr_image, MDP.ROI)

    ph = pft.get_histogram(mask)
    ch = cft.get_histogram(mask)

    similarity = cft.compare_histogram(ph, ch)
    scene_changed = similarity >= MDP.SCENE_CHANGE_THRESHOLD

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

    # curr_roi_frame = ImageUtil.crop(curr_image, MDP.ROI)

    tracker.init(curr_image, init_bbox)

    retry_count = 0
    while not event_stop_thread.wait(0.0001):
        try:
            display_data = in_queue.popleft()
            frame = display_data.input_image
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count + 1}')
            if retry_count == 5:
                break

            retry_count += 1
            time.sleep(0.5)
            continue

        retry_count = 0

        display_data.roi_frame = ImageUtil.crop(display_data.input_image, MDP.ROI)
        (success, box) = tracker.update(display_data.roi_frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            x = x + MDP.ROI[0]
            y = y + MDP.ROI[1]
            logging.info(f'tracking: {x}, {y}, {w}, {h}')

            display_data.object_image = copy.copy(display_data.input_image)
            cv2.rectangle(display_data.object_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(display_data.object_image, (MDP.ROI[0], MDP.ROI[1]),
                          (MDP.ROI[2], MDP.ROI[3]),
                          (220, 220, 220), 2)

            display_queue.append(display_data)
            out_queue.put(display_data)
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
    maxlen = 20
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
            display_data = q.popleft()
        except IndexError:
            logging.info(f'No frame left in in_queue : retrying {retry_count + 1}')
            if retry_count == 5:
                break
            retry_count += 1
            time.sleep(0.5)
            continue

        if is_first:
            prev_frame = display_data.input_image
            is_first = False
            continue

        # if nskip % 5 == 0:
        # mse=((frame[hidx, widx[:nsamples], :] - prev_frame[hidx[:nsamples], widx[:nsamples], :]) ** 2).mean(axis=None)
        mse = ((display_data.input_image[hidx, widx, :] - prev_frame[hidx, widx, :]) ** 2).mean(axis=None)
        dq.append(mse)
        prev_frame = display_data.input_image
        # logging.info('fetch frame')
        # logging.info(f'dqsize: {len(dq)}/{maxlen}')
        if len(dq) == maxlen:
            v = np.var(dq)
            if v < 30.0:
                stability += 1
            else:
                stability = 0

            if stability > 10:
                logging.info(f'Scene stabilized: var={v:.2f} stability={stability}')
                return
        # nskip+=1

        if q_window_name:
            display_queue.append(display_data)


def del_slice(q, s, e):
    for i in range(e - 1, s - 1, -1):
        del q[i]

    return q


def track_object_presence(obj_queue):

    # Tracking Analysis.ipynb 참조
    object_status = [x[0] for x in obj_queue]
    # if MDP.DEBUG:
    #     logging.info(f'object_status : {"".join(str(x) for x in object_status)}')

    smoothed_object_status, T = ImageUtil.smooth(object_status, MDP.MOVING_WINDOW_SIZE)
    n = len(smoothed_object_status)
    frame_margin = int(round(1. / T + 0.5))

    # if MDP.DEBUG:
    #     curr_index = obj_queue[-1][1].index
    #     filename = f'{MDP.output_dir}/{curr_index:08d}-tracking'

    idx0 = np.argwhere(smoothed_object_status[::-1] < T)
    tracking = np.sum(idx0[0:frame_margin]) == np.sum(range(frame_margin))
    # logging.info(f'Track = {tracking}')

    if not tracking:
        # if MDP.DEBUG:
        #     # plt.close(fig)
        #     fd.close()
        return

    s0 = None
    e0 = None
    # 1st step : 연속적으로 T 이상인 index를 찾자
    idx = np.argwhere(smoothed_object_status >= T)
    if len(idx) == 0:
        # del object_status[0:n-frame_margin]
        del_slice(obj_queue, 0, n - frame_margin)
        # print(f'연속적으로 T 이상인 구간이 없음 {object_status}')
    else:
        s0 = int(idx[0])
        e0 = int(idx[-1])
        # print(f'연속적으로 T 이상인 구간 {s0} {e0}')

    s1 = None
    e1 = None
    if s0 is not None and e0 is not None:
        # 2nd step : s0, e0 에서 frame_margin 만큼 확장하자
        # 2-1 : s0-frame_margin >= 0   # s1 = s0-frame_margin
        # 2-2 : s0-frame_margin < 0    # s1 = max(0, s0-frame_margin)
        s1 = max(0, s0 - frame_margin)
        # 2-3 : e0+frame_margin < n    # e1 = e0+frame_margin
        if e0 + frame_margin < n:
            e1 = e0 + frame_margin
        elif n - s0 >= MDP.MAX_OBJECT_SLICE:
            e1 = n
        else:
            # 2-4 ; e0+frame_margin >= n    # e1 = None (다음 state 가 1이될 가능성이 있으므로 기다린다)
            e1 = None

    object_slice = None
    # step#3 : 0 부터 e1 까지 지운다
    # if len([x is None for x in [s0, e0, s1, e1]])==0:
    # print(f'{s0}, {e0}, {s1}, {e1}')
    if [s0, e0, s1, e1].count(None) < 1:
        object_slice = [obj_queue[i][1] for i in range(s1, e1)]
        del_slice(obj_queue, 0, e1)
        # del object_status[:e1]
        # smoothed_object_status = np.delete(smoothed_object_status, np.s_[:e1])

        # object_status
    # else:
    # print(f'None included')
    # if MDP.DEBUG:
    #     n = len(smoothed_object_status)
    #     x = range(n)
    #     ax[2].plot(x, smoothed_object_status)
    #     ax[2].plot(x, np.ones(n) * T)
    #     ax[2].plot(x, np.array(object_status), 'o')
    #     ax[2].set_title('Remaining object sequence')
        # print(f'object_status : {object_status}')

    if object_slice:
        # logging.info(f'To be write : { " ".join([str(x.index) for x in object_slice])}')
        q = deque()
        for x in object_slice:
            make_overlay_image(x)
            q.append(x.object_image)
        filename = object_slice[0].time.replace(microsecond=0).isoformat() + '.mp4'
        filename = FileUtil.make_valid_filename(filename)
        if len(q) < 70:
            filename = FileUtil.make_suffix_filename(filename, '-suspicious')
        logging.info(f'Writing file : {len(q)} frames to {filename} with {MDP.FPS:.1f} fps.')
        th = VideoFileWritingThread(name=f'VideoFileWritingThread',
                                    args=(q, 'Object detected', filename, MDP.FPS))
        th.start()

        if MDP.DEBUG:
            logging.info(f'object_slice : \n[{",".join("{:.2f} ".format(x) for x in smoothed_object_status[s1:e1])}]')

    #     if MDP.DEBUG:
    #         x = range(s1, e1)
    #         n = len(smoothed_object_status[s1:e1])
    #         ax[1].set_title('Sliced object sequence')
    #         ax[1].plot(x, smoothed_object_status[s1:e1])
    #         ax[1].plot(x, np.ones(n)*T)
    #         ax[1].plot(x, np.array(object_status[s1:e1]), 'o')
    #         # print(f'object_status : {object_status}')
    #         fig.savefig(f'{MDP.output_dir}/{curr_index:08d-tracking.png')
    #
    # if MDP.DEBUG:
    #     plt.close(fig)


def monitor_thread(in_queue, obj_list, mask):
    default_learning_rate = 0.1

    logging.info(f'Started')
    logging.info(f'output dir : {MDP.output_dir}')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    fgbg.setBackgroundRatio(default_learning_rate)
    # automatically chosen
    learning_rate = -1
    retry_count = 0
    prev_image = None
    curr_image = None
    # wait until event is set,
    latency = 0
    while event_monitor.wait():
        if event_stop_thread.wait(0.0001):
            obj_list.append(None)
            break

        if not event_capture_ready.wait(60.0):
            logging.info(f'Waiting for capturing time out')
            obj_list.append(None)
            break

        try:
            display_data = in_queue.popleft()
            latency = (datetime.now() - display_data.time).seconds
            if display_data.index % 100 == 0:
                logging.info(f'Monitoring latency : {latency:.1f}')
        except IndexError:
            if retry_count == 10:
                logging.info(f'No frame left in in_queue : tried {retry_count} times')
                obj_list.append(None)
                break
            retry_count += 1
            time.sleep(1)
            continue

        retry_count = 0

        display_data.roi_image = ImageUtil.crop(display_data.input_image, MDP.ROI)
        display_data.roi_image = cv2.cvtColor(display_data.roi_image, cv2.COLOR_BGR2GRAY)
        # not good result
        # roi_frame = cv2.equalizeHist(roi_frame)
        display_data.roi_image = cv2.GaussianBlur(display_data.roi_image, None, 3)

        if prev_image is None:
            prev_image = display_data.roi_image
        else:
            prev_image = curr_image
        curr_image = display_data.roi_image

        # scene change detection. input color should be preserved
        scene_changed, similarity = detect_scene_change(prev_image, curr_image, mask)
        if scene_changed:
            logging.info(f'scene_changed {display_data.index:04d}: {similarity}')
            wait_for_scene_stable(in_queue, display_data.input_image.shape[1], display_data.input_image.shape[0], None)
            learning_rate = 1
            continue

        if learning_rate == 1:
            logging.info(f'Re-learning scene from now {display_data.index:04d}')
            fgbg.apply(curr_image, None, 1)
            learning_rate = default_learning_rate

        fgmask = fgbg.apply(curr_image, None, learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        display_data.model_image = fgbg.getBackgroundImage()
        display_data.fg_image = fgmask
        display_data.object_image = copy.copy(display_data.input_image)

        # TODO: remove debugging code
        # obj_size=(0, ImageUtil.width(ROI)*ImageUtil.height(ROI))
        obj_size = MDP.OBJECT_SIZE
        x, y, w, h, s, con = find_object(display_data.fg_image, obj_size)
        if w is not None:
            # logging.info(f'Object detected')

            # METHOD1 : Using tracker
            # th = threading.Thread(None, tracking_thread, "tracking_thread",
            #                       args=(curr_image, in_queue, (x, y, w, h), obj_list))
            # th.start()
            # event_tracking.set()
            # # pause me
            # event_monitor.clear()

            # METHOD2 : Just print out
            x = x + MDP.ROI[0]
            y = y + MDP.ROI[1]
            # logging.info(f'tracking: {x}, {y}, {w}, {h}')

            cv2.rectangle(display_data.object_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(display_data.object_image, (MDP.ROI[0], MDP.ROI[1]), (MDP.ROI[2], MDP.ROI[3]),
                          (220, 220, 220), 2)
            cv2.drawContours(display_data.object_image, [con + [MDP.ROI[0], MDP.ROI[1]]], 0, (0, 0, 255), thickness=3)
            cv2.putText(display_data.fg_image, f'contourArea {display_data.index:04d}: {s}',
                        (0, display_data.fg_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255))
            # logging.info(f'Object size {display_data.index:04d}: {s}')

        object_list.append((1 if w is not None else 0, display_data))
        track_object_presence(object_list)
        display_queue.append(display_data)

    logging.info(f'Stopped')

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


def read_param(argv):
    # name of video file (eg. video.avi)
    # or image sequence (eg. img_%02d.jpg, which will read samples like
    # img_00.jpg, img_01.jpg, img_02.jpg, ...)
    # or URL of video stream (eg.
    # protocol://host:port/script_name?script_params|auth) or GStreamer pipeline string in gst-launch tool format in
    # case if GStreamer is used as backend Note that each video stream or IP camera feed has its own URL scheme.
    # Please refer to the documentation of source stream to know the right URL.
    MDP.video_source = argv[1]
    if not os.path.exists(MDP.video_source):
        # logging.info(f'File not found : {MDP.video_source}')
        MDP.output_dir = f'data/MotionDetector-{DateUtil.get_current_timestring()}'
    else:
        MDP.output_dir = os.path.join(os.path.dirname(MDP.video_source),
                                      os.path.splitext(os.path.basename(MDP.video_source))[0])
    if MDP.DEBUG_FRAME_TO_FILE:
        try:
            plt.ioff()
            os.mkdir(MDP.output_dir)
        except FileExistsError:
            pass
        finally:
            if not os.path.exists(MDP.output_dir):
                raise RuntimeError(f"Cannot create directory : {MDP.output_dir}")


def writing_display_image_thread(q, output_dir):
    logging.info(f'Started')
    while True:
        if not event_capture_ready.wait():
            logging.info(f'Waiting for capturing time out')
            break

        display_data = q.get()
        if display_data is None:
            break

        if not os.path.exists(output_dir):
            logging.info(f'Directory not exists : {output_dir}')
            continue

        cv2.imwrite(f'{output_dir}/{display_data.index:08d}-model.png', display_data.model_image)
        cv2.imwrite(f'{output_dir}/{display_data.index:08d}-foreground.png', display_data.fg_image)
        cv2.imwrite(f'{output_dir}/{display_data.index:08d}-object.png', display_data.object_image)
    logging.info(f'Finished')


def make_overlay_image(display_data):
    if display_data.object_image is None or display_data.fg_image is None or display_data.model_image is None:
        return

    dx, dy = (20, 20)
    line_width = 5
    fg_image_color = cv2.cvtColor(display_data.fg_image, cv2.COLOR_GRAY2BGR)
    model_image_color = cv2.cvtColor(display_data.model_image, cv2.COLOR_GRAY2BGR)
    display_data.object_image, nw, nh = ImageUtil.overlay_image(copy.copy(display_data.object_image),
                                                                fg_image_color,
                                                                dx, dy,
                                                                int(display_data.object_image.shape[1] * 0.25))
    cv2.rectangle(display_data.object_image, (dx, dy), (dx+nw, dy+nh), (255, 0, 255), line_width)
    display_data.object_image, nw, nh = ImageUtil.overlay_image(copy.copy(display_data.object_image),
                                                                model_image_color,
                                                                dx, nh + 2*dy,
                                                                int(display_data.object_image.shape[1] * 0.25))
    cv2.rectangle(display_data.object_image, (dx, nh + 2*dy), (nw + dx, 2*(nh + dy)), (0, 255, 255), line_width)


def main_thread():

    read_param(sys.argv)

    read_video_params(MDP.video_source)

    ImageUtil.create_image_window("source", MDP.VIDEO_WIDTH, MDP.VIDEO_HEIGHT,
                                  ImageUtil.width(MDP.ROI), ImageUtil.height(MDP.ROI), 0.5)

    mask = ImageLoader.read_image('data/mask1.png')
    mask = imutils.resize(mask, MDP.VIDEO_WIDTH)
    th_capture = threading.Thread(None, capture_thread, "capture_thread",
                                  args=(MDP.video_source, input_queue, MDP.NUM_SKIP_FRAME, MDP.FRAME_SCALE))
    th_monitor = threading.Thread(None, monitor_thread, "monitor_thread",
                                  args=(input_queue, object_list, mask))
    th_display = threading.Thread(None, writing_display_image_thread, "display_image_writing_thread",
                                  args=(file_queue, MDP.output_dir))

    # start capture
    th_capture.start()

    # start monitor thread
    event_monitor.set()
    th_monitor.start()

    th_display.start()

    while True:
        q_len = len(display_queue)
        if event_stop_thread.wait(0.0001) and q_len < 1:
            file_queue.put(None)
            break

        if q_len > 0:
            display_data = display_queue.popleft()
            # logging.info(f'Updating image')
            if display_data.input_image is not None:
                cv2.imshow('source', display_data.input_image)
            if display_data.model_image is not None:
                cv2.imshow('Model', display_data.model_image)
            if display_data.fg_image is not None:
                cv2.imshow('Difference', display_data.fg_image)

            make_overlay_image(display_data)
            if display_data.object_image is not None:
                cv2.imshow('Detector', display_data.object_image)

            if MDP.DEBUG_FRAME_TO_FILE:
                file_queue.put(display_data)

        if cv2.waitKey(50) == ord('q'):
            file_queue.put(None)
            break

    event_monitor.set()
    event_stop_thread.set()

    logging.info(f'Stopped')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main_thread()
