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


def green_channel(image):
    image = image[:, :, 1].reshape(
        image.shape[0],
        image.shape[1],
        1)
    return image


def plot_image_patches(x, ksize_rows=64, ksize_cols=64):
    nr = x.shape[1]
    nc = x.shape[2]
    fig = plt.figure()
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.01, hspace=0.01)

    for i in range(nr):
        for j in range(nc):
            ax = plt.subplot(gs[i*nc+j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('auto')
            plt.imshow(x[0, i, j].reshape(ksize_rows, ksize_cols, 3))
    return fig


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
