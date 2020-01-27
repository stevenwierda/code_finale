# **** Pytorch transforms for numpy ****
# https://github.com/mlagunas/pytorch-nptransforms

from __future__ import division
import random
import numpy as np
from math import floor

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

def crop(img, target, parameters, dims = 3):
    w, h, th, tw = parameters
    if dims == 3:
        crop_sample = img[:, h:h + th, w:w + tw], target[h:h + th, w:w + tw]
    elif dims == 4:
        crop_sample = img[:, :, h:h + th, w:w + tw], target[:, h:h + th, w:w + tw]
    else:
        crop_sample = img, target
    return crop_sample

def get_params(h,w, output_size):
    # read crop size
    th, tw = output_size, output_size

    # get crop indexes
    i = random.randint(0, w - tw)
    j = random.randint(0, h - th)

    return i, j, th, tw

def cropping_iterations(shape, args):
    # shape ... C:H:W
    horizontal_fit = floor(shape[-1] / args.crop_size[0])
    vertical_fit = floor(shape[-2] / args.crop_size[1])
    return args.subsamples * min(vertical_fit, horizontal_fit)