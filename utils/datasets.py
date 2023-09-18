# YOLOv5 dataset utils and dataloaders

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from engine.transforms.default import build_transform
test_transform = build_transform('none')
# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=224):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.img_size = img_size
        self.files = images
        self.nf = ni# number of files
        self.mode = 'image'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

         # Read image
        self.count += 1
        # img0 = cv2.imread(path)  # BGR
        # img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        img0 = cv2.imread(path)
        assert img0 is not None, 'Image Not Found ' + path
        # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        # img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # print(type(img))
        # img = np.ascontiguousarray(img)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = test_transform(img.convert("RGB"))
        # Convert
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=224):
        self.img_size = img_size
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        # cv2.CAP_DSHOW 可以在释放摄像头后不报错
        self.cap = cv2.VideoCapture(eval(pipe), cv2.CAP_DSHOW) if pipe.isnumeric() \
            else cv2.VideoCapture(pipe) # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        # print(f'webcam {self.count}: ', end='')

        # Padded resize

        # img = letterbox(img0, self.img_size, stride=16)[0]
        #
        # # Convert
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        img = test_transform(img)

        return img_path, img, img0, self.cap

    def __len__(self):
        return 0

# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized








