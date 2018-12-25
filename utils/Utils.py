import os
import argparse


def normalize(self, nparray, alpha=-1):
    nparray = np.array(nparray)
    nparray[2, 2] = alpha
    nparray = np.ma.array(nparray, mask=False)
    nparray.mask[2, 2] = True
    sumation = nparray.sum()
    nparray = nparray/sumation
    nparray = np.array(nparray)
    return nparray


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
