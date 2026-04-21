# Inference for ONXX model
import cv2
cuda = True
m = "yolov-tiny.onxx"
img = cv2.imread('horses.jpg')

import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(m, providers=providers)
