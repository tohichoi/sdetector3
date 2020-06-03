import copy
import time

import imutils
import cv2
import numpy as np
import os
from scipy.special import kl_div
import matplotlib.pyplot as plt


class GlobalData:
    ROI = [404, 0, 1006, 680]


class ImageSaver:
    def __init__self(self):
        pass

    @staticmethod
    def write_image_to_video(filename, image_list, scale=1.0, fps=10):
        if len(image_list) < 1:
            return False

        width = int(image_list[0].shape[1] * scale + 0.5)
        height = int(image_list[0].shape[0] * scale + 0.5)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vcap_out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for f in image_list:
            vcap_out.write(f)

        vcap_out.release()

        return True


class ImageLoader:
    def __init__self(self):
        pass

    @staticmethod
    def preprocess(image, *args):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = imutils.resize(image, width=image.shape[1] * 0.5)
        return image

    @staticmethod
    def read_image(filepath, preprocess=None, preprocess_args=None):
        image = cv2.imread(filepath)
        if preprocess:
            image = preprocess(image, *preprocess_args)
        return image

    @staticmethod
    def read_filelist(filename):
        filelist = []
        with open(filename) as fd:
            for line in fd.readlines():
                filename = line.strip()
                if len(filename) < 1:
                    continue

                if not os.path.exists(filename):
                    print(f'{filename} : not exists')
                    continue

                filelist.append(filename)

        return filelist

    @staticmethod
    def read_image_from_filelist(filename, preprop_function=None, preprop_function_args=None):
        images = []
        filelist = ImageLoader.read_filelist(filename)
        for filename in filelist:
            img = ImageLoader.read_image(filename)
            if preprop_function:
                img = preprop_function(img, *preprop_function_args)
            images.append(img)

        return images, filelist


class VideoUtil:
    @staticmethod
    def get_size(source):
        vcap = cv2.VideoCapture(source)
        w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vcap.release()
        return w, h

    @staticmethod
    def estimate_fps(video_source, num_frames=120):
        # Start default camera
        video = cv2.VideoCapture(video_source)

        cv2_fps = video.get(cv2.CAP_PROP_FPS)

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = video.read()

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start

        # Calculate frames per second
        fps = float(num_frames) / seconds

        # Release video
        video.release()

        return fps, cv2_fps


class ImageUtil:
    @staticmethod
    def width(roi):
        return roi[2] - roi[0]

    @staticmethod
    def height(roi):
        return roi[3] - roi[1]

    @staticmethod
    def area(roi):
        return ImageUtil.width(roi) * ImageUtil.height(roi)

    @staticmethod
    def coord(roi):
        x = roi[0]
        y = roi[1]
        w = ImageUtil.width(roi)
        h = ImageUtil.height(roi)

        return x, y, w, h

    @staticmethod
    def crop(frame, roi):
        x, y, w, h = ImageUtil.coord(roi)
        return frame[y:y + h, x:x + w]

    @staticmethod
    def get_mask_image(frame, roi):
        mask_bg = np.ones_like(frame) * 255
        mask_bg[roi[1]:roi[3], roi[0]:roi[2]] = 0
        mask = mask_bg.astype('uint8')
        return mask

    @staticmethod
    def show_image(image, waitkey=50):
        title = 'temp'
        cv2.imshow(title, image)
        cv2.waitKey(waitkey)
        cv2.destroyWindow(title)
        cv2.waitKey(waitkey)

    @staticmethod
    def overlay_image(baseimage, overlay, x, y, w):
        bw = baseimage.shape[1]
        bh = baseimage.shape[0]

        # sanity check
        # newimage = copy.copy(baseimage)
        newimage = baseimage
        newoverlay = imutils.resize(overlay, w)
        nw = newoverlay.shape[1]
        nh = newoverlay.shape[0]
        newimage[y:y + nh, x:x + nw] = newoverlay

        return newimage, nw, nh

    @staticmethod
    def create_image_window(source, w, h, rw, rh, s):
        scale = s
        wm, hm = (5, 35)
        nw, nh = (int(w * scale), int(h * scale))
        rnw, rnh = (int(rw * scale), int(rh * scale))
        wn = [source, 'Detector', 'Model', 'Difference', 'Threshold']

        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO

        cv2.namedWindow(wn[0], flags)
        cv2.moveWindow(wn[0], 0, 0)
        cv2.resizeWindow(wn[0], nw, nh)

        cv2.namedWindow(wn[1], flags)
        cv2.moveWindow(wn[1], nw + wm, 0)
        cv2.resizeWindow(wn[1], nw, nh)

        cv2.namedWindow(wn[2], flags)
        cv2.moveWindow(wn[2], 0, nh + hm)
        cv2.resizeWindow(wn[2], rnw, rnh)

        cv2.namedWindow(wn[3], flags)
        cv2.moveWindow(wn[3], rnw + wm, nh + hm)
        cv2.resizeWindow(wn[3], rnw, rnh)

        cv2.namedWindow(wn[4], flags)
        cv2.moveWindow(wn[4], 2 * (rnw + wm), nh + hm)
        cv2.resizeWindow(wn[4], rnw, rnh)

    # s: 1d signal
    @staticmethod
    def smooth(s, window_size=5):
        if len(s) < 1:
            return None, None

        wlen = min(window_size, len(s))
        T = 1. / wlen
        w = np.ones(wlen)
        status = np.convolve(w / w.sum(), s, mode='same')

        return status, T


class FeatureExtractor:
    def __init__(self, image, roi, filename=None):
        self.image = image
        self.filename = filename
        self.features = {}
        self.roi = roi

    # test
    # data=norm.rvs(10.0, 2.5, size=500)
    # show_distribution(data, norm.pdf, *(10.0, 2.5))
    # data=skewnorm.rvs(10.0, 2.5, size=500)
    # show_distribution(data, skewnorm.pdf, *(10.0, 2.5))
    @staticmethod
    def show_distribution(self, data, pdf, color, *args):
        plt.hist(data, bins=100, density=True, alpha=0.6, color=color)

        n = len(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, n)
        p = pdf(x, *args)
        plt.plot(x, p, 'k', linewidth=2)
        # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        # plt.title(title)

    def get_color_variance(self, nsamples):
        w = self.image.shape[1]
        h = self.image.shape[0]

        widx = np.random.randint(w, size=nsamples)
        hidx = np.random.randint(h, size=nsamples)

        v = np.sum(np.diff(self.image[hidx, widx, :].astype(np.int16)), axis=1).flatten()

        self.features['color_variance'] = v

        return v

    def is_color(self, nsamples):
        stat = self.get_color_variance(nsamples)
        r = float(len(list(filter(lambda x: x < 5, stat)))) / nsamples
        is_color = r > 0.98
        self.features['color'] = is_color

        return is_color

    def get_histogram(self, mask_image=None):
        frame2 = self.image
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        # frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
        frame2 = cv2.medianBlur(frame2, 5)

        # cv2.equalizeHist(frame2)
        mask = None
        if mask_image is not None:
            mask = mask
        elif self.roi is not None:
            mask_bg = np.ones_like(frame2) * 255
            mask_bg[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = 0
            mask = mask_bg.astype('uint8')

        hist = cv2.calcHist([frame2], [0], mask, [256], [0, 256])
        # hist.resize(hist.size)
        hist += 1
        hist /= hist.sum()
        # hist = hist[128:]

        self.features['histogram_pdf'] = np.transpose(hist)

        return self.features['histogram_pdf']

    @staticmethod
    def compare_histogram(hista, histb):
        kld = np.sum(kl_div(hista, histb))

        return kld

    def compare_histogram1(self, other_histogram_pdf):
        if self.image is None:
            return 0

        kld = np.sum(kl_div(self.features['histogram_pdf'], other_histogram_pdf))

        return kld

    def __str__(self):
        return f'{self.filename}, {str(self.features.keys)}'
