import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from random import shuffle
from skimage.transform import resize

path = 'dataset-dist/phase-01/training/pristine/'

images = os.listdir(path)


def convert_to_jpeg(image, qf):
    cv2.imwrite('img_CV2.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    imj = cv2.imread('img_CV2.jpg')
    imj = imj[:(imj.shape[0]//128)*128, :(imj.shape[1]//128)*128, 1]
    os.remove('img_CV2.jpg')
    return imj


def resampling_image(image, scale, n, m):
    tmprs = resize(
        img[:, :, 1],
        (n*scale, m*scale),
        mode='reflect',
        anti_aliasing=True)
    i1_start = (tmprs.shape[0]-n)//2
    i1_stop = i1_start + n
    i2_start = (tmprs.shape[1]-m)//2
    i2_stop = i2_start + m
    tmprs = tmprs[i1_start:i1_stop, i2_start:i2_stop]
    # Make sure the dimension is right
    tmprs = tmprs[:(tmprs.shape[0]//128)*128, :(tmprs.shape[1]//128)*128]
    return tmprs


def median_filtering_image(image, k_size):
    return cv2.medianBlur(image, k_size)


def gaussian_blurring_image(image, k_size, std):
    return cv2.GaussianBlur(image, (k_size, k_size), std)


def medianfilter_gaussianblurring(image, k_size, std, qf):
    mf_im = median_filtering_image(image, k_size)
    mf_gb_im = gaussian_blurring_image(mf_im, k_size, std)
    mf_gb_jp_im = convert_to_jpeg(mf_gb_im, qf)
    return mf_gb_jp_im


def gaussianblurring_medianfilter(image, k_size, std, qf):
    gb_im = gaussian_blurring_image(image, k_size, std)
    gb_mf_im = median_filtering_image(gb_im, k_size)
    gb_mf_jp_im = convert_to_jpeg(gb_mf_im, qf)
    return gb_mf_jp_im


def medianfilter_resampling(image, k_size, scale, n, m, qf):
    mf_im = median_filtering_image(image, k_size)
    mf_rs_im = resampling_image(mf_im, scale, n, m)
    mf_rs_jp_im = convert_to_jpeg(mf_rs_im, qf)
    return mf_rs_jp_im


def resampling_medianfilter(image, k_size, scale, n, m, qf):
    rs_im = resampling_image(image, scale, n, m)
    rs_mf_im = median_filtering_image(rs_im, k_size)
    rs_mf_jp_im = convert_to_jpeg(rs_mf_im, qf)
    return rs_mf_jp_im


def gaussianblurring_resampling(image, k_size, std, scale, n, m, qf):
    gb_im = gaussian_blurring_image(image, k_size, std)
    gb_rs_im = resampling_image(gb_im, scale, n, m)
    gb_rs_jp_im = convert_to_jpeg(gb_rs_im, qf)
    return gb_rs_jp_im


def resampling_gaussianblurring(image, k_size, std, scale, n, m, qf):
    rs_im = resampling_image(image, scale, n, m)
    rs_gb_im = gaussian_blurring_image(rs_im, k_size, std)
    rs_gb_jp_im = convert_to_jpeg(rs_gb_im, qf)
    return rs_gb_jp_im


for image in images:
    img = cv2.imread(image)

    n = 640
    m = 640

    X = np.zeros((25000, 128, 128, 1), dtype=np.uint8)
    y = np.zeros(25000, dtype=np.int64)
    count = 0
    k = 0

    i1_start = (img.shape[0]-n)//2
    i1_stop = i1_start + n
    i2_start = (img.shape[1]-m)//2
    i2_stop = i2_start + m
    img = img[i1_start:i1_stop, i2_start:i2_stop, :]

    # OR
    tmp = img[:(img.shape[0]//128)*128, :(img.shape[1]//128)*128, 1]
    # RS
    tmprs = resampling_image(tmp, scale=1.5, n=n, m=m)
    # MF
    tmpm = median_filtering_image(tmp, k_size=5)
    # GB
    tmpg = gaussian_blurring_image(tmp, k_size=5, std=1.1)
    # MF_GB
    tmpmgb = medianfilter_gaussianblurring(tmp, 5, std=1.1, 90)
    # GB_MF
    tmpgbm = gaussianblurring_medianfilter(tmp, 5, std=1.1, 90)
    # MF_RS
    tmpmrs = medianfilter_resampling(tmp, 5, scale=1.5, n, m, 90)
    # RS_MF
    tmprsm = resampling_medianfilter(tmp, 5, scale=1.5, n, m, 90)
    # GB_RS
    tmpgbrs = gaussianblurring_resampling(tmp, 5, std=1.1, scale=1.5, n, m, 90)
    # RS_GB
    tmprsgb = resampling_gaussianblurring(tmp, 5, std=1.1, scale=1.5, n, m, 90)

    vblocks = np.vsplit(tmp, tmp.shape[0]/128)
    shuffle(vblocks)
    # Make sure the dimension is right
    vblocks = vblocks[:len(vblocks)]
    imcount = 0

    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 0
            imcount += 1
            if count == 50000:
                    break
            count += 1
        if count == 50000:
            break
    print ('OR ' + str(imcount))
    if count == 50000:
        break

    vblocks = np.vsplit(tmpm, tmpm.shape[0]/128)
    shuffle(vblocks)
    vblocks = vblocks[:len(vblocks)]
    imcount = 0
    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 1
            imcount += 1
            if count == 50000:
                break
            count += 1
        if count == 50000:
            break
    print ('MF ' + str(imcount))
    if count == 50000:
        break

    vblocks = np.vsplit(tmpg, tmpg.shape[0]/128)
    shuffle(vblocks)
    vblocks = vblocks[:len(vblocks)]
    imcount = 0
    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 2
            imcount += 1
            if count == 50000:
                break
            count += 1
        if count == 50000:
            break
    print ('GB ' + str(imcount))
    if count == 50000:
        break

    vblocks = np.vsplit(tmprs, tmprs.shape[0]/128)
    shuffle(vblocks)
    vblocks = vblocks[:len(vblocks)]
    imcount = 0
    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 4
            imcount += 1
            if count == 50000:
                break
            count += 1
        if count == 50000:
            break
    print ('RS ' + str(imcount))
    if count == 50000:
        break

    vblocks = np.vsplit(tmpj, tmpj.shape[0]/128)
    shuffle(vblocks)
    vblocks = vblocks[:len(vblocks)]
    imcount = 0
    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 5
            imcount += 1
            if count == 50000:
                break
            count += 1
        if count == 50000:
            break
    print ('JPG ' + str(imcount))
    if count == 50000:
        break
