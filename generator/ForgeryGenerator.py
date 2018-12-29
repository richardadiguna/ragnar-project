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

    # Save patch to JPEG with QF 70
    cv2.imwrite(
        'img_CV2_70.jpg',
        img[:, :, 1],
        [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    # Resampling or Resizing with scale
    tmprs = resize(
        img[:, :, 1],
        (640*1.5, 640*1.5),
        mode='reflect',
        anti_aliasing=True)
    i1_start = (tmprs.shape[0]-n)//2
    i1_stop = i1_start + n
    i2_start = (tmprs.shape[1]-m)//2
    i2_stop = i2_start + m
    tmprs = tmprs[i1_start:i1_stop, i2_start:i2_stop]
    # Make sure the dimension is right
    tmprs = tmprs[:(tmprs.shape[0]//128)*128, :(tmprs.shape[1]//128)*128]

    tmp = img[:(img.shape[0]//128)*128, :(img.shape[1]//128)*128, 1]

    # JPEG Compression
    tmpj = cv2.imread('img_CV2_70.jpg')
    # Make sure the dimension is right
    tmpj = tmpj[:(tmpj.shape[0]//128)*128, :(tmpj.shape[1]//128)*128, 1]

    os.remove('img_CV2_70.jpg')

    # Median Filtering
    tmpm = cv2.medianBlur(tmp, 5)

    # Gaussian Blurring
    tmpg = cv2.GaussianBlur(tmp, (5, 5), 0)

    # Additive Gaussian Noise
    awgn = 2.0*np.random.randn(tmp.shape[0], tmp.shape[1])
    tmpw = tmp+awgn
    tmpw = np.clip(tmpw, 0, 255)
    tmpw = tmpw.astype(np.uint8)

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

    vblocks = np.vsplit(tmpw, tmpw.shape[0]/128)
    shuffle(vblocks)
    vblocks = vblocks[:len(vblocks)]
    imcount = 0
    for v in vblocks:
        hblocks = np.hsplit(v, v.shape[1]/128)
        shuffle(hblocks)
        hblocks = hblocks[:len(hblocks)]
        for h in hblocks:
            X[count-1] = h.reshape((1, 128, 128, 1))
            y[count-1] = 3
            imcount += 1
            if count == 50000:
                break
            count += 1
        if count == 50000:
            break
    print ('WGN ' + str(imcount))
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
