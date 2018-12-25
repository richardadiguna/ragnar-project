import os
import math
import random
import numpy as np
import tensorflow as tf
import ImageManipulation

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


def patches_generator(folder, patch_size, stride, T=500):
    images_fname = os.listdir(folder)

    for name in images_fname:
        file_path = os.path.join(folder, name)
        fname = file_path[:-4][31:]

        image = np.array(Image.open(file_path))

        patches = tf_patch_extract(image, patch_size, stride)
        rand_index = np.random.choice(patches.shape[0], T)
        patches = patches[rand_index]

        for index in range(patches.shape[0]):
            patch = patches[index]
            image = Image.fromarray(patch.astype(np.uint8))

            path = 'positive_patches/' + fname + '_' + str(index) + '.jpg'

            image.save(path, 'JPEG', quality=100)


def get_all_images(folder):
    images = []
    images_fname = os.listdir(folder)
    for name in images_fname:
        file_path = os.path.join(folder, name)
        image = np.array(Image.open(file_path))
        images.append(image)
    images = np.array(images, dtype=np.float32)
    return images


def tampered_images_generation(folder, param_dict,
                               target_dir, manipulation=True):
    images = get_all_images(folder)
    counter = 0
    if manipulation:
        while counter < images.shape[0]:
            for key in param_dict.keys():
                for value in param_dict[key]:
                    if counter < images.shape[0]:
                        image = images[counter]
                        manipulated_im = \
                            ImageManipulation.manipulate_image(
                                image, key, param_val=value, counter=counter)
                        if key != 'jpeg_compression':
                            path = target_dir + str(counter) + '.jpg'
                            save_patch(path, manipulated_im)
                        counter += 1


def save_patch(path, im_patch):
    image = Image.fromarray(im_patch.astype(np.uint8))
    image.save(path, 'JPEG', quality=95)


def save_raw_image_to_jpeg(folder, save_dir):
    fnames = os.listdir(folder)
    for fname in fnames:
        path = os.path.join(folder, fname)
        im = Image.open(path)
        name = fname[:-4] + '.jpg'
        save_path = os.path.join(save_dir, name)
        im.save(save_path, 'JPEG', quality=100)
