# encoding: utf-8

import re
import os
import sys
import threading
import time
import copy
import datetime
from queue import Queue, Empty
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
from collections import deque, defaultdict
from telegram import Bot
from telegram.error import NetworkError, Unauthorized, RetryAfter
import maya


class MotionDetectionParam:
    FRAME_SCALE = 1
    # Contour area (WIDTH X HEIGHT)
    OBJECT_SIZE = (200 * 100, 200 * 400)
    NUM_SKIP_FRAME = 0
    NUM_SKIP_DISPLAY_FRAME = 2
    ROI = [404, 0, 1070, 680]
    SCENE_CHANGE_THRESHOLD = 0.4
    MAX_OBJECT_SLICE = 300
    MOVING_WINDOW_SIZE = 30
    REPORT_SECONDS = 60 * 60
    VIDEO_ORG_WIDTH = -1
    VIDEO_ORG_HEIGHT = -1
    # scaled width
    VIDEO_WIDTH = -1
    # scaled height
    VIDEO_HEIGHT = -1
    # reporting_time = (datetime.time(hour=7, minute=0, second=0), datetime.time(hour=23, minute=59, second=59))
    reporting_time = (datetime.time(hour=6, minute=30, second=0), datetime.time(hour=2, minute=0, second=0))
    video_source = None
    output_dir = None
    # 0 : Do not write
    # 1 : Write input when detected
    # 2 : Write all
    DEBUG_FRAME_TO_FILE = 0
    FPS = None
    show_window_flag = [False, True, False, False, False]
    # 1 : Send all
    # 2 : Send confident video
    # 3 : Do not send
    send_video = 1
    mask = None
    mask_area = None


class TelegramData:
    video_q = Queue()
    bot = None
    CHAT_ID = None
    TOKEN = None


MDP = MotionDetectionParam
TD = TelegramData

input_queue = deque()
display_queue = deque()
object_list = deque()
file_queue = Queue()
message_queue = Queue()
last_frame_queue = Queue()

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
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0 - c) * v


class DisplayData:
    def __init__(self, index=0, timestamp=0):
        self.index = index
        self.time = timestamp if timestamp > 0 else datetime.datetime.now()
        self.input_image = None
        self.model_image = None
        self.fg_image = None
        self.object_image = None
        self.roi_image = None


class ObjectShape:
    valid = 0
    x = None
    y = None
    w = None
    h = None
    s = None
    c = None
    i = None


def notify_alive_thread(last_frame_q, message_q):
    # while not event_stop_thread.wait(60*60):
    #     image = last_frame_queue.get()
    #     filename = MDP.output_dir + 'last_frame.png'
    #     cv2.imwrite(filename, image)
    logging.info('Started')

    st = maya.parse(MDP.reporting_time[0])
    et = maya.parse(MDP.reporting_time[1])
    if et.hour < st.hour:
        et = et + datetime.timedelta(days=1)

    logging.info(f'Reporting hour : {st.iso8601():}h to {et.iso8601()}h')
    while not event_stop_thread.wait(0):
        logging.info(f'processing queue')
        try:
            image = last_frame_q.get(True, MDP.REPORT_SECONDS * 0.5)
            if image is None:
                logging.info(f'Exit message received')
                break
        except Empty:
            continue

        nt = maya.parse(datetime.datetime.now())
        # logging.info(f'Start time: {st}\nEnd time: {et}\nNow: {now}')

        if not (st.hour <= nt.hour < et.hour):
            # logging.info(f'Start time: {st}\nEnd time: {et}\nNow: {now}')
            logging.info("Notification is off")
            continue

        filename = MDP.output_dir + 'last_frame.png'
        cv2.imwrite(filename, image)

        retry_count = 0
        while retry_count < 10:
            try:
                logging.info(f'sending alive message')
                TelegramData.bot.send_photo(chat_id=TelegramData.CHAT_ID,
                                            photo=open(filename, 'rb'),
                                            disable_notification=True,
                                            timeout=30,
                                            caption='싸라있네!')
                logging.info(f'alive message sent')
                break

            except NetworkError:
                logging.info('Exception : NetworkError')
                retry_count += 1
                time.sleep(10)
                continue

            except RetryAfter:
                logging.info('Exception : RetryAfter')
                retry_count += 1
                time.sleep(10)
                continue

    logging.info('Finished')


def send_message_thread(message_q):
    logging.info('Started')

    while True:
        message = message_q.get()
        if message is None:
            break

        retry_count = 0
        while retry_count < 10:
            try:
                TelegramData.bot.send_message(chat_id=TelegramData.CHAT_ID, text=message)
                break

            except NetworkError:
                logging.info('Exception : NetworkError')
                retry_count += 1
                time.sleep(10)
                continue

            except RetryAfter:
                logging.info('Exception : RetryAfter')
                retry_count += 1
                time.sleep(10)
                continue

    logging.info('Finished')


def send_video_thread(writing_thread_handle, filename, logprob):
    writing_thread_handle.join(30)
    if writing_thread_handle.is_alive():
        logging.info(f'{writing_thread_handle.name} is still running. Aborting message sending.')
        return

    logging.info('Started')
    fn = os.path.basename(filename)
    logger.info(f'{filename}')
    # d=datetime.datetime.fromtimestamp(v.createdtime, datetime.timezone.utc)
    msg = f'순심이 확률(log): {logprob:.2f}'

    retry_count = 0
    while retry_count < 1:
        try:
            TelegramData.bot.send_message(chat_id=TelegramData.CHAT_ID, text=msg)
            TelegramData.bot.send_video(chat_id=TelegramData.CHAT_ID,
                                        caption=fn,
                                        video=open(filename, 'rb'),
                                        timeout=120)
            break

        except NetworkError:
            logging.info('Exception : NetworkError')
            retry_count += 1
            time.sleep(10)
            continue

        except RetryAfter:
            logging.info('Exception : RetryAfter')
            retry_count += 1
            time.sleep(10)
            continue

    logging.info('Finished')


def send_video(writing_thread_handle, filename, logprob):
    # q=Config.tg_video_q
    # while not q.empty():
    #     v=q.get()
    th = threading.Thread(None, send_video_thread, "send_video_thread",
                          (writing_thread_handle, filename, logprob))

    th.start()


def capture_thread(video_source, input_q, last_frame_q, nskipframe, frame_scale):
    logging.info(f'Started')

    logging.info(f'Connecting to device {video_source} ...')
    vcap = cv2.VideoCapture(video_source)
    logging.info('Connected.')

    logging.info(f'Estimating FPS')
    fps, hw_fps = VideoUtil.estimate_fps(vcap)
    logging.info(f'FPS : {hw_fps:.1f}(device), {fps:.1f}(estimated)')
    # MDP.FPS = min(hw_fps, fps)
    MDP.FPS = fps

    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    event_capture_ready.set()

    last_check_time = datetime.datetime.now()
    frame_index = 0
    while not event_stop_thread.wait(0):
        ret, frame = vcap.read()
        if not ret:
            logging.info('Video decoding error occurred.')
            break

        if frame is None or len(frame) < 1:
            logging.info('Device has no more frames')
            break

        frame_index += 1
        if nskipframe > 0 and frame_index % (nskipframe + 1) != 0:
            continue

        if frame_scale != 1:
            frame = imutils.resize(frame, int(w * frame_scale))

        # cv2.putText(frame, f'{frame_index-1}', (0, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))
        ImageUtil.put_text(frame, f'{frame_index - 1}', MDP.ROI[2] + 20, frame.shape[0] - 50,
                           (0, 0, 0), None, 1*MDP.FRAME_SCALE, 1)

        display_data = DisplayData(index=frame_index - 1)
        display_data.input_image = frame
        input_q.append(display_data)

        dt = display_data.time - last_check_time
        # logging.info(f'report seconds: {dt.seconds}')
        if dt.seconds > MDP.REPORT_SECONDS:
            last_check_time = display_data.time
            last_frame_q.put(copy.copy(display_data.input_image))

        # time.sleep(fps / 1000)

    event_stop_thread.set()
    vcap.release()
    logging.info(f'Finished')


def detect_scene_change(prev_image, curr_image, mask):
    pft = FeatureExtractor(prev_image, MDP.ROI)
    cft = FeatureExtractor(curr_image, MDP.ROI)

    ph = pft.get_histogram(mask)
    ch = cft.get_histogram(mask)

    similarity = cft.compare_histogram(ph, ch)
    scene_changed = similarity >= MDP.SCENE_CHANGE_THRESHOLD

    return scene_changed, similarity


def tracking_thread(curr_image, input_q, init_bbox, output_q):
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
    while not event_stop_thread.wait(0):
        try:
            display_data = input_q.popleft()
            frame = display_data.input_image
            retry_count = 0
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
            output_q.put(display_data)
        else:
            logging.info(f'object lost')
            # resume monitor thread
            break

    event_monitor.set()

    logging.info(f'Finished')


def identify_object(display_data, contours, object_size_threshold):
    o = ObjectShape()
    o.x, o.y, o.w, o.h = cv2.boundingRect(contours[0])
    o.s = cv2.contourArea(contours[0])
    # logging.info(f'contourArea : {s}')
    o.c = contours[0]
    o.i = cv2.mean(display_data.roi_image, mask=display_data.fg_image)[0]
    classify = {}

    roiw = ImageUtil.width(MDP.ROI)
    roih = ImageUtil.height(MDP.ROI)

    classify['size'] = (1 if object_size_threshold[1] > o.s >= object_size_threshold[0] else 0, o.s)
    classify['intensity'] = (1 if o.i > 130 else 0, o.i)

    # classify['width'] = (1 if roiw * 0.2 < w <= roiw * 0.8 else 0, w)
    # classify['height'] = (1 if roih * 0.2 < h <= roih * 0.7 else 0, h)

    # logging.info(f'contourArea : {s}, min: {object_size_threshold[0]}, max: {object_size_threshold[1]}')
    # center of mass feature
    # logging.info(f'CX={CX}')
    # logging.info(f'CY={CY}')

    if len(classify) == np.sum([x[0] for x in classify.values()]):
        o.valid = 1
        return o

    return o


def find_object(display_data, object_size_threshold):
    lbound = 240
    ubound = 255
    ncon = 0

    # mask = cv2.inRange(fgmask, lbound, ubound)
    _, mask = cv2.threshold(display_data.fg_image, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ncon = len(contours)
    if ncon < 1:
        return ObjectShape()

    # method#1 : contour bounding box
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # center of mass feature
    # CX=[]
    # CY=[]
    # for c in contours:
    #     M = cv2.moments(c)
    #     cx = np.NaN if M['m00'] == 0 else int(M['m10'] / M['m00'])
    #     cy = np.NaN if M['m00'] == 0 else int(M['m01'] / M['m00'])
    #     CX.append(cx)
    #     CY.append(cy)

    return identify_object(display_data, contours, object_size_threshold)


def show_status_text(image, text, line_index=0):
    # ImageUtil.put_text(image, "Stability learning", image.shape[1] - 200,
    #                    100, (0xff, 0, 0), (0, 0, 0), 1, cv2.LINE_AA)
    ImageUtil.put_text(image, text, MDP.ROI[2] + 20,
                       70 * (line_index + 1), (0, 0, 0xff), (0, 0, 0), 0.7*MDP.FRAME_SCALE, 1)


def wait_for_scene_stable(input_q, w, h, q_window_name=None):
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

    last_frame_number = 0
    retry_count = 0
    # nskip=1
    while True:
        try:
            display_data = input_q.popleft()
            if display_data is None:
                break
            last_frame_number = display_data.index
            retry_count = 0
        except IndexError:
            if retry_count == 10:
                logging.info(
                    f'No frame left in in_queue (last frame : {last_frame_number}) : retrying {retry_count + 1}')
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

        display_data.object_image = copy.copy(display_data.input_image)
        show_status_text(display_data.object_image, "Stabilizing")

        # logging.info('fetch frame')
        # logging.info(f'dqsize: {len(dq)}/{maxlen}')
        if len(dq) == maxlen:
            v = np.var(dq)
            if v < 30.0:
                stability += 1
            else:
                stability = 0

            if stability > 10:
                logging.info(f'Scene stabilized: var={v:.2f}')
                return
        # nskip+=1

        if q_window_name:
            display_queue.append(display_data)


def del_slice(q, s, e):
    for i in range(e - 1, s - 1, -1):
        del q[i]

    return q


def get_object_slice_prob(smoothed_object_status):
    # 0 이상인 값을 추출한다
    s = smoothed_object_status

    # empty 리스트이면 prob = 0.0
    return np.sum(np.log(s[np.where(s > 0)[0]]))


def write_object_slice(object_q, smoothed_object_status):
    q = deque()
    for x in object_q:
        make_overlay_image(x)
        q.append(x.object_image)

    filename = object_q[0].time.replace(microsecond=0).isoformat() + '.mp4'
    filename = FileUtil.make_valid_filename(filename)

    confidence_level = 1 if len(q) < 70 else 2

    if confidence_level == 1:
        filename = FileUtil.make_suffix_filename(filename, '-suspicious')

    filepath = MDP.output_dir + filename
    logging.info(f'Writing file from {object_q[0].index}: {len(q)} frames to {filepath} with {MDP.FPS:.1f} fps.')
    logprob = get_object_slice_prob(smoothed_object_status)
    th = VideoFileWritingThread(name=f'VideoFileWritingThread',
                                args=(q, f'{logprob:.2f}', filepath, MDP.FPS, MDP.FRAME_SCALE))
    th.start()

    logging.info(
        f'object_slice (log prob = {logprob:.2f}): \n[{",".join("{:.2f} ".format(x) for x in smoothed_object_status)}]')

    if confidence_level >= MDP.send_video:
        send_video(th, filepath, logprob)
    else:
        logging.info(f'Confidence level ={confidence_level}, send_video value={MDP.send_video}')


def track_object_presence(object_q):
    # Tracking Analysis.ipynb 참조
    object_status = [x[0].valid for x in object_q]
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
        del_slice(object_q, 0, n - frame_margin)
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
    object_size_slice = None
    object_intensity_slice = None
    # step#3 : 0 부터 e1 까지 지운다
    # if len([x is None for x in [s0, e0, s1, e1]])==0:
    # print(f'{s0}, {e0}, {s1}, {e1}')
    if [s0, e0, s1, e1].count(None) < 1:
        object_slice = [object_q[i][1] for i in range(s1, e1)]
        object_size_slice = [object_q[i][0].s for i in range(s1, e1)]
        object_intensity_slice = [object_q[i][0].i for i in range(s1, e1)]
        del_slice(object_q, 0, e1)
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
        # 일정 크기 이상인 영역이 포함되어있으면 reject
        reject_condition = {}
        s = []
        i = []
        if object_size_slice:
            s = [x for x in object_size_slice if x is not None and x > 0.5 * MDP.mask_area]
            reject_condition['size'] = len(s) > 0

        if object_intensity_slice:
            i = [x for x in object_intensity_slice if x is not None and x < 90]
            reject_condition['intensity'] = len(i) > 0

        if len([x for x in reject_condition.values() if x is True]) >= 1:
            logging.info(f'Dropping object slice due to abnormal area : {s}')
            logging.info(f'Dropping object slice due to abnormal intensity : {i}')
        else:
            # logging.info(f'To be write : { " ".join([str(x.index) for x in object_slice])}')
            write_object_slice(object_slice, smoothed_object_status[s1:e1])

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


def show_latency(latency):
    logging.info(f'Monitoring latency : {latency:.1f}')


def monitor_thread(input_q, obj_list, mask):
    # // default parameters of gaussian background detection algorithm
    # static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
    # static const float defaultVarThreshold2 = 4.0f*4.0f;
    # static const int defaultNMixtures2 = 5; // maximal number of Gaussians in mixture
    # static const float defaultBackgroundRatio2 = 0.9f; // threshold sum of weights for background test
    # static const float defaultVarThresholdGen2 = 3.0f*3.0f;
    # static const float defaultVarInit2 = 15.0f; // initial variance for new components
    # static const float defaultVarMax2 = 5*defaultVarInit2;
    # static const float defaultVarMin2 = 4.0f;
    #
    # // additional parameters
    # static const float defaultfCT2 = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components
    # static const unsigned char defaultnShadowDetection2 = (unsigned char)127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
    # static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation

    # learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    # nframes : apply 가 호출될 때마다 증가
    # 초당 10 프레임이고 history가 100 이면? 10초
    #  learning rate = 1/100 = 0.01
    # 20초
    history_frame = 1000
    default_learning_rate = 1. / history_frame

    logging.info(f'Started')
    logging.info(f'output dir : {MDP.output_dir}')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, detectShadows=True)
    # fgbg = cv2.createBackgroundSubtractorKNN()
    fgbg.setVarThreshold(16)
    # fgbg.setNMixtures(5)
    # 0.8: 223 frame 이 지난후 background
    # 0.7: 356 frame 이 지난후 background
    fgbg.setBackgroundRatio(0.6)
    # fgbg.setHistory(300)
    # fgbg.setShadowThreshold(0.8)
    # automatically chosen
    learning_rate = -1
    retry_count = 0
    loop_count = 0
    prev_image = None
    curr_image = None
    # wait until event is set,
    latency = 0

    mask2 = ImageUtil.crop(mask, MDP.ROI)

    while not event_stop_thread.wait(0):
        if not event_capture_ready.wait(120.0):
            logging.info(f'Capturing is not ready in 120 seconds.')
            break

        try:
            display_data = input_q.popleft()
            latency = (datetime.datetime.now() - display_data.time).seconds
            loop_count += 1
            if loop_count % 1000 == 0:
                logging.info(f'Monitoring latency : {latency:.1f}')
                loop_count = 0
        except IndexError:
            if retry_count == 12:
                logging.info(f'No frame in in_queue : tried {retry_count} times')
                break
            # else:
            #     logging.info(f'No frame in in_queue : retrying {retry_count}')
            retry_count += 1
            time.sleep(10)
            continue

        retry_count = 0

        display_data.roi_image = ImageUtil.crop(display_data.input_image, MDP.ROI)
        display_data.roi_image = cv2.cvtColor(display_data.roi_image, cv2.COLOR_BGR2GRAY)
        # not good result
        # roi_frame = cv2.equalizeHist(roi_frame)
        display_data.roi_image = cv2.GaussianBlur(display_data.roi_image, None, 3)
        display_data.roi_image = ImageUtil.get_mask_roi_image(display_data.roi_image, mask2)

        if prev_image is None:
            prev_image = display_data.roi_image
        else:
            prev_image = curr_image
        curr_image = display_data.roi_image

        # scene change detection. input color should be preserved
        scene_changed, similarity = detect_scene_change(prev_image, curr_image, mask)
        if scene_changed:
            logging.info(f'Scene changed {display_data.index:04d}: KL Divergence = {similarity:.2f}')
            wait_for_scene_stable(input_q, display_data.input_image.shape[1], display_data.input_image.shape[0],
                                  'source')
            object_list.clear()
            logging.info(f'Object list is cleared')
            learning_rate = 1
            continue

        if learning_rate == 1:
            logging.info(f'Re-learning scene and detecting object from now {display_data.index:04d}')
            fgbg.apply(curr_image, None, 1)
            learning_rate = default_learning_rate
            # 51 frame
            # fgbg.setBackgroundRatio(0.95)

        fgmask = fgbg.apply(curr_image, None, learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        display_data.model_image = fgbg.getBackgroundImage()
        display_data.fg_image = fgmask if display_data.index > 10 else np.zeros_like(fgmask)
        display_data.object_image = copy.copy(display_data.input_image)
        show_status_text(display_data.object_image, "Detecting")

        # TODO: remove debugging code
        # obj_size=(0, ImageUtil.width(ROI)*ImageUtil.height(ROI))
        obj_size = MDP.OBJECT_SIZE
        o = find_object(display_data, obj_size)
        if o.c is not None:
            x = o.x + MDP.ROI[0]
            y = o.y + MDP.ROI[1]
            # logging.info(f'tracking: {x}, {y}, {w}, {h}')

            cv2.rectangle(display_data.object_image, (x, y), (x + o.w, y + o.h), (255, 255, 0), 2)
            cv2.rectangle(display_data.object_image, (MDP.ROI[0], MDP.ROI[1]), (MDP.ROI[2], MDP.ROI[3]),
                          (220, 220, 220), 2)
            cv2.drawContours(display_data.object_image, [o.c + [MDP.ROI[0], MDP.ROI[1]]], 0, (0, 0, 255), thickness=3)
            cv2.putText(display_data.fg_image, f'S: {o.s:.0f}, I: {o.i:.0f}',
                        (0, display_data.fg_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 64, 0))

        object_list.append((o, display_data))
        track_object_presence(object_list)
        display_queue.append(display_data)

    obj_list.append(None)

    event_stop_thread.set()

    logging.info(f'Finished')


def read_video_params(vsrc):
    vcap = cv2.VideoCapture(vsrc)

    logging.info(f'Waiting for opening device {vsrc}')
    ev = threading.Event()
    ev.clear()
    retry = 0
    while not ev.wait(1):
        if vcap.isOpened():
            logging.info('Device is opened')
            break
        elif retry == 10:
            logging.info('Cannot open device')
            raise Exception()

        retry += 1

    MDP.VIDEO_ORG_WIDTH = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    MDP.VIDEO_ORG_HEIGHT = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    MDP.VIDEO_WIDTH = int(MDP.VIDEO_ORG_WIDTH * MDP.FRAME_SCALE + 0.5)
    MDP.VIDEO_HEIGHT = int(MDP.VIDEO_ORG_HEIGHT * MDP.FRAME_SCALE + 0.5)
    MDP.ROI = (MDP.FRAME_SCALE * np.array(MDP.ROI)).astype(np.int).tolist()
    MDP.OBJECT_SIZE = (MDP.FRAME_SCALE * np.array(MDP.OBJECT_SIZE)).astype(np.int).tolist()

    vcap.release()

    filename = 'data/mask1.png'
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    
    mask = ImageLoader.read_image(filename)
    mask = imutils.resize(mask, MDP.VIDEO_WIDTH)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    MDP.mask = mask
    MDP.mask_area = cv2.contourArea(contours[0])


def read_param(argv):
    # name of video file (eg. video.avi)
    # or image sequence (eg. img_%02d.jpg, which will read samples like
    # img_00.jpg, img_01.jpg, img_02.jpg, ...)
    # or URL of video stream (eg.
    # protocol://host:port/script_name?script_params|auth) or GStreamer pipeline string in gst-launch tool format in
    # case if GStreamer is used as backend Note that each video stream or IP camera feed has its own URL scheme.
    # Please refer to the documentation of source stream to know the right URL.
    MDP.video_source = argv[1]
    # if not os.path.exists(MDP.video_source):
    #     MDP.output_dir = f'log/MotionDetector-{DateUtil.get_current_timestring()}/'
    # else:
    #     MDP.output_dir = os.path.join(os.path.dirname(MDP.video_source),
    #                                   os.path.splitext(os.path.basename(MDP.video_source))[0]) + '/'
    # if MDP.DEBUG_FRAME_TO_FILE:
    MDP.output_dir = f'log/MotionDetector-{DateUtil.get_current_timestring()}/'

    try:
        plt.ioff()
        os.makedirs(MDP.output_dir)
    except FileExistsError:
        pass
    finally:
        if not os.path.exists(MDP.output_dir):
            raise RuntimeError(f"Cannot create directory : {MDP.output_dir}")

    telegram_config = 'config/telegram.json'
    if not os.path.exists(telegram_config):
        raise RuntimeError(f"Cannot find directory : {telegram_config}")

    with open(telegram_config) as fd:
        cf = json.load(fd)
        TelegramData.CHAT_ID = cf['bot_chatid']
        TelegramData.TOKEN = cf['bot_token']
        TelegramData.bot = Bot(TelegramData.TOKEN)

    if os.uname()[1] == 'raspberrypi':
        MDP.FRAME_SCALE = 0.5
        # Contour area (WIDTH X HEIGHT)
        MDP.NUM_SKIP_FRAME = 1
        MDP.NUM_SKIP_DISPLAY_FRAME = 1
 

def write_display_image_thread(input_q, output_dir):
    logging.info(f'Started')
    nsamples = 1000

    widx = None
    hidx = None
    prev_frame = None

    while True:
        if not event_capture_ready.wait():
            logging.info(f'Waiting for capturing time out')
            break

        display_data = input_q.get()
        if display_data is None:
            break

        # if widx is None:
        widx = np.random.randint(display_data.input_image.shape[1], size=nsamples)
        # if hidx is None:
        hidx = np.random.randint(display_data.input_image.shape[0], size=nsamples)
        if prev_frame is None:
            prev_frame = np.zeros_like(display_data.input_image)

        if not os.path.exists(output_dir):
            logging.info(f'Directory not exists : {output_dir}')
            break

        mse = ((display_data.input_image[hidx, widx, :] - prev_frame[hidx, widx, :]) ** 2).mean(axis=None)
        write_condition = mse > 20
        # logging.info(f'mse: {mse}')

        if write_condition:
            if MDP.DEBUG_FRAME_TO_FILE >= 1:
                cv2.imwrite(f'{output_dir}/{display_data.index:08d}-input.png', display_data.input_image)

            if MDP.DEBUG_FRAME_TO_FILE >= 2:
                cv2.imwrite(f'{output_dir}/{display_data.index:08d}-model.png', display_data.model_image)
                cv2.imwrite(f'{output_dir}/{display_data.index:08d}-foreground.png', display_data.fg_image)
                cv2.imwrite(f'{output_dir}/{display_data.index:08d}-object.png', display_data.object_image)

        prev_frame = copy.copy(display_data.input_image)

    logging.info(f'Finished')


def make_overlay_image(display_data):
    if display_data.object_image is None or display_data.fg_image is None or display_data.model_image is None:
        return

    iw = display_data.object_image.shape[1]
    ih = display_data.object_image.shape[0]
    rw = display_data.model_image.shape[1]
    rh = display_data.model_image.shape[0]

    dx, dy = (20, 20)
    line_width = 2
    hr = 0.5 * (ih - dy * 3) * 0.5 / ih
    rw = hr

    # fg_image_color = cv2.cvtColor(display_data.fg_image, cv2.COLOR_GRAY2BGR)
    fg_image_color = cv2.applyColorMap(display_data.fg_image, cv2.COLORMAP_BONE)

    # model_image_color = cv2.cvtColor(display_data.model_image, cv2.COLOR_GRAY2BGR)
    model_image_color = cv2.applyColorMap(display_data.model_image, cv2.COLORMAP_RAINBOW)
    display_data.object_image, nw, nh = ImageUtil.overlay_image(copy.copy(display_data.object_image),
                                                                fg_image_color,
                                                                dx, dy,
                                                                int(display_data.object_image.shape[1] * rw))
    cv2.rectangle(display_data.object_image, (dx, dy), (dx + nw, dy + nh), (64, 64, 64), line_width)
    display_data.object_image, nw, nh = ImageUtil.overlay_image(copy.copy(display_data.object_image),
                                                                model_image_color,
                                                                dx, nh + 2 * dy,
                                                                int(display_data.object_image.shape[1] * rw))
    cv2.rectangle(display_data.object_image, (dx, nh + 2 * dy), (nw + dx, 2 * (nh + dy)), (64, 64, 64), line_width)


def main_thread():
    read_param(sys.argv)

    logging.info(f'Reading video parameters from {MDP.video_source}')

    read_video_params(MDP.video_source)

    ImageUtil.create_image_window("source", MDP.VIDEO_WIDTH, MDP.VIDEO_HEIGHT,
                                  ImageUtil.width(MDP.ROI), ImageUtil.height(MDP.ROI), 1,
                                  show_flag=MDP.show_window_flag)

    th_capture = threading.Thread(None, capture_thread, "capture_thread",
                                  args=(
                                  MDP.video_source, input_queue, last_frame_queue, MDP.NUM_SKIP_FRAME, MDP.FRAME_SCALE))
    th_monitor = threading.Thread(None, monitor_thread, "monitor_thread",
                                  args=(input_queue, object_list, MDP.mask))
    th_write_file = threading.Thread(None, write_display_image_thread, "write_display_image_thread",
                                     args=(file_queue, MDP.output_dir))
    th_send_message = threading.Thread(None, send_message_thread, "send_message_thread", args=(message_queue,))

    th_notify_alive = threading.Thread(None, notify_alive_thread, "notify_alive_thread",
                                       args=(last_frame_queue, message_queue))

    # start capture
    th_capture.start()
    th_monitor.start()
    th_write_file.start()
    th_send_message.start()
    th_notify_alive.start()

    frame_index = 0
    while True:
        q_len = len(display_queue)
        if event_stop_thread.wait(0):
            break

        if q_len > 0:
            display_data = display_queue.popleft()

            frame_index += 1
            if MDP.NUM_SKIP_DISPLAY_FRAME > 0 and frame_index % (MDP.NUM_SKIP_DISPLAY_FRAME + 1) != 0:
                continue

            # logging.info(f'Updating image')
            if display_data.input_image is not None and MDP.show_window_flag[2]:
                cv2.imshow('source', display_data.input_image)
            if display_data.model_image is not None and MDP.show_window_flag[2]:
                cv2.imshow('Model', display_data.model_image)
            if display_data.fg_image is not None and MDP.show_window_flag[3]:
                cv2.imshow('Difference', display_data.fg_image)

            make_overlay_image(display_data)
            if display_data.object_image is not None:
                cv2.imshow('Detector', display_data.object_image)

            if MDP.DEBUG_FRAME_TO_FILE > 0:
                file_queue.put(display_data)

        if cv2.waitKey(50) == ord('q'):
            event_stop_thread.set()
            break

    event_monitor.clear()

    file_queue.put(None)
    last_frame_queue.put(None)
    message_queue.put('순탐이 종료!')
    message_queue.put(None)

    logging.info(f'Finished')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main_thread()
