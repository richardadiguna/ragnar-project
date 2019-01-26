import os
import math
import random
import numpy as np
import tensorflow as tf

from PIL import Image


def patch_extract(image, patch_size, stride):
    (L1, L2, L3) = image.shape
    patch_num_L1 = int(math.floor((L1-patch_size)/stride)+1)
    patch_num_L2 = int(math.floor((L2-patch_size)/stride)+1)
    patches_num = patch_num_L1 * patch_num_L2
    patches = np.zeros(
        (patches_num, patch_size, patch_size, L3),
        dtype=np.float32)

    start_l1 = 0
    end_l1 = 0
    start_l2 = 0
    end_l2 = 0
    patches_num_real = 0
    for l1 in range(0, patch_num_L1):
        for l2 in range(0, patch_num_L2):
            start_l1 = (l1)*stride
            end_l1 = start_l1 + patch_size
            start_l2 = (l2)*stride
            end_l2 = start_l2 + patch_size
            if end_l1 <= L1 and end_l2 <= L2:
                patch = image[start_l1:end_l1, start_l2:end_l2, :]
                if patches_num_real < patches_num:
                    patches[patches_num_real, :, :, :] = patch
                    patches_num_real = patches_num_real + 1

    return patches, patch_num_L1, patch_num_L2


def map_to_full(feamap, patch_num_L1, patch_num_L2, image, patch_size, stride):
    (L1, L2, L3) = image.shape
    feamap_full = np.zeros((L1, L2, 1), dtype=float)
    feamap_full_num = np.zeros((L1, L2, 1), dtype=float)
    start_l1 = 0
    end_l1 = 0
    start_l2 = 0
    end_l2 = 0
    for l1 in range(0, (patch_num_L1)):
        for l2 in range(0, (patch_num_L2)):
            start_l1 = (l1)*stride
            end_l1 = start_l1 + patch_size
            start_l2 = (l2) * stride
            end_l2 = start_l2 + patch_size
            if end_l1 <= L1 and end_l2 <= L2:
                feamap_full[
                    start_l1:end_l1,
                    start_l2:end_l2, :] = \
                        feamap_full[
                            start_l1:end_l1,
                            start_l2:end_l2, :] + feamap[l1, l2]
                feamap_full_num[start_l1:end_l1, start_l2:end_l2, :] = \
                    feamap_full_num[start_l1:end_l1, start_l2:end_l2, :] + 1
    o_l = np.where(feamap_full_num=0)
    feamap_full[o_l] = 1.0
    feamap_full_num[o_l] = 1.0
    feamap_full = feamap_full/feamap_full_num
    if end_l1 < L1:
        for l1 in range((end_l1), L1):
            feamap_full[l1, :] = feamap_full[end_l1-1, :]
    if end_l2 < L2:
        for l2 in range((end_l2), L2):
            feamap_full[:, l2] = feamap_full[:, end_l2 - 1]
    return feamap_full


def tf_patch_extract(image, patch_size, stride):
    tf.reset_default_graph()

    x = tf.placeholder(
        tf.float32,
        shape=[1, image.shape[0], image.shape[1], image.shape[2]])

    patches = tf.extract_image_patches(
        x, [1, patch_size, patch_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        padding='VALID')

    samples = tf.squeeze(patches, [0])
    num_sample = samples.get_shape()[0] * samples.get_shape()[1]

    image_patches = tf.reshape(
        samples,
        [num_sample, patch_size, patch_size, 3])

    with tf.Session() as sess:
        sample_patches = sess.run(
            image_patches,
            feed_dict={x: np.array([image])})

    return sample_patches
