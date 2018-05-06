# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

#import matplotlib as mpl
#mpl.use('TkAgg')

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os


def main(_):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)

    im_width = 1024
    im_height = 1024

    # read satellite in_data
    data_dir = FLAGS.valid_dir
    if FLAGS.data != "valid":
        data_dir = FLAGS.test_dir + FLAGS.data

    if FLAGS.data != "valid":
        name_number = np.load(FLAGS.name_dir + FLAGS.data + ".npy")
    else:
        name_number = np.load(FLAGS.name_dir + FLAGS.data + "_" + FLAGS.side_length + ".npy")

    num = len(name_number)

    prediction = np.load("prediction_" + FLAGS.data + "_" + str(FLAGS.prediction_random_num) + ".npy")

    merged_np = np.zeros((im_height, im_width * 2, 3))

    image_pholder = tf.placeholder(tf.float32, shape=[im_height, im_width, 3])
    convert_gray = tf.image.rgb_to_grayscale(image_pholder)

    with tf.Session() as sess:
        for idx, number in enumerate(name_number):
            in_name = str(number) + "_sat.jpg"
            out_name = str(number) + "_mask.png"

            in_img = mpimg.imread(join(data_dir, in_name)) #(1024, 1024, 3)

            merged_np[:, 0:1024, :] = in_img  #(1024, 1024, 3)

            if FLAGS.data == 'valid':
                out_img = mpimg.imread(join(data_dir, out_name))[:, :, 0]  #(1024, 1024)   (0 or 1)
                merged_np[:, 1024:, 2] = out_img * 255  #(1024, 1024) blue

            if FLAGS.out_label:
                out_img = mpimg.imread((join(FLAGS.label_dir, out_name)))[:, :, 0]
                merged_np[:, 1024:, 2] = out_img * 255

            gray_img = sess.run(convert_gray, feed_dict={image_pholder: in_img})  #(1024, 1024, 1)

            not_prediction = 1 - prediction[idx, :, :, :]

            multiply_gray_img = np.multiply(gray_img, not_prediction)  #(1024, 1024, 1)

            merged_np[:, 1024:, 1] = multiply_gray_img.reshape(im_height, im_width) # green

            merged_np[:, 1024:, 0] = (prediction[idx, :, :, :] * 255).reshape(im_height, im_width) #(1024, 1024) red   0 or 1

            merged_np = merged_np.astype(np.uint8)

            plt.imsave("./%d/%s/%d.png" % (FLAGS.prediction_random_num, FLAGS.data, int(number)), merged_np)


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('gpu_id', 3, 'id ranges from 0 to 3')
    tf.app.flags.DEFINE_integer('prediction_random_num', 3, 'map to the prediction')
    tf.app.flags.DEFINE_string('data', 'final_test', 'valid or dev_test or final_test')
    tf.app.flags.DEFINE_boolean('out_label', True, 'whether to use out labels')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'number of images per mini-batch')
    tf.app.flags.DEFINE_string('test_dir', '/media/workspace/bgong/data/road-extraction/', 'test data directory')
    tf.app.flags.DEFINE_string('valid_dir', '/media/workspace/bgong/data/road-extraction/train', 'validation data directory')

    tf.app.flags.DEFINE_string('label_dir', '/home/chenny1229/label_from_website/', 'label directory')
    tf.app.flags.DEFINE_string('name_dir', '/home/chenny1229/parameters/512/names/', 'name directory')
    # name_dir need to be changed to bgong (cp names to bgong directory)
    # label_dir also needs to be changed to bgong

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
