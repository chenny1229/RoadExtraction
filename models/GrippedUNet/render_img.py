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
    data_dir = FLAGS.data_dir
    name_number = np.load(FLAGS.name_number)
    num = len(name_number)

    prediction = np.load(FLAGS.prediction)

    #print(np.any(prediction[:] == 255))

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
                out_img = mpimg.imread((join(FLAGS.out_label_dir, out_name)))[:, :, 0]
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
    tf.app.flags.DEFINE_integer('gpu_id', 1, 'id ranges from 0 to 3')
    tf.app.flags.DEFINE_integer('prediction_random_num', 5, 'map to the prediction')
    tf.app.flags.DEFINE_string('data', 'test', 'validation or test')
    tf.app.flags.DEFINE_boolean('out_label', True, 'whether to use out labels')
    tf.app.flags.DEFINE_string('name_number', 'dev_test_name_number.npy', 'the number names of data')
    tf.app.flags.DEFINE_string('prediction', 'prediction_test_5.npy', 'prediction file')
    tf.app.flags.DEFINE_string('data_dir', '/media/workspace/bgong/data/road-extraction/dev_test', 'test or validation data directory')
    tf.app.flags.DEFINE_string('out_label_dir', '', 'outside label directory')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'number of images per mini-batch')

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
