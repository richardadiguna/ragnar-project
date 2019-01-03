import os
import argparse
import numpy as np


def normalize(kernel, alpha=-1):
    kernel_arr = np.array(kernel)
    kernel_arr[2, 2] = alpha
    kernel_arr = np.ma.array(kernel_arr, mask=False)
    kernel_arr.mask[2, 2] = True
    sumation = kernel_arr.sum()
    kernel_arr = kernel_arr / sumation
    kernel_arr = np.array(kernel_arr)
    return kernel_arr


def green_channel(im_path, window_size=128, stride=128):
    image = cv2.imread(im_path)
    image = image[:, :, 1].reshape(
        image.shape[0],
        image.shape[1],
        1)
    return image


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exist(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating driectories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args
