import logging
import os
import re
import threading
import time

import cv2
import numpy as np

from image import ImageUtil
from mydateutil import DateUtil


def create_logging():
    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class FileUtil:
    @staticmethod
    def make_symlink(filepath, dname):
        fn = os.path.basename(filepath)
        f, e = os.path.splitext(fn)
        nf = f + '-grayscale' + e
        path = os.path.join(dname, nf)
        # cv2.imwrite(path, train_images[j].astype('uint8'))
        # shutil.copy(filelist[j], dname)
        s = os.path.relpath(filepath, dname)
        d = os.path.join(dname, fn)
        try:
            os.symlink(s, d)
        except OSError as e:
            return False

        return True

    @staticmethod
    def merge_images_to_video(self, folder, dest_filename, fourcc='mp4v'):
        for dirpath, dirnames, filenames in os.walk(folder):
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
            vcap_out = cv2.VideoWriter(dest_filename, fourcc, 20.0, (1280, 720))
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                frame = cv2.imread(filepath)
                # frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame_gray=cv2.GaussianBlur(frame_gray, (3, 3), 0)
                vcap_out.write(frame)
            vcap_out.release()

    @staticmethod
    def replace_extension(filename, rep_ext):
        return os.path.splitext(filename)[0] + rep_ext

    @staticmethod
    def make_valid_filename(filename, replace_string=''):
        s = re.sub('[-:]', replace_string, filename)
        return s

    @staticmethod
    def make_suffix_filename(filename, suffix):
        f, e = os.path.splitext(filename)
        return f + suffix + e


class VideoFileWritingThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(VideoFileWritingThread, self).__init__()
        self.target = target
        self.name = name
        self.q = args[0]
        self.info = args[1]
        self.filename = args[2]
        self.fps = min(15, args[3])
        self.text_scale = args[4]

    def run(self):
        n = len(self.q)
        w = self.q[0].shape[1]
        h = self.q[0].shape[0]

        # logging.info(f'{threading.get_ident()} write_framebuf started : {self.q.qsize()}')
        # s = DateUtil.get_current_timestring()
        fn = FileUtil.replace_extension(self.filename, ".mp4")

        logging.info(f'Writing {fn}')
        # fps = self.__estimate_fps()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vcap_out = cv2.VideoWriter(fn, fourcc, self.fps, (w, h))

        while len(self.q) > 0:
            frame = self.q.popleft()
            ImageUtil.put_text(frame, self.info, frame.shape[1] - 150, frame.shape[0] - 120,
                               (0, 0, 0), (0xff, 0xcd, 0xd2), 0.7*self.text_scale, 2)
            vcap_out.write(frame)
            time.sleep(0.05)
            # logging.info(f'{threading.get_ident()} : {self.q.qsize()}')

        vcap_out.release()
        # write_event.clear()
        # logging.info(f'{threading.get_ident()} write_framebuf finished')

        # vf=VideoFile(fn, end_msec)
        # Config.tg_video_q.put_nowait(fn)


if __name__ == 'fileutil':
    create_logging()
