# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import matplotlib.image as mpimg
from layer_funs import gripped_unet

#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import os

def validate_graph(logits, ground_truth, num_classes, threshold):

    # pixel_predict = tf.reshape(tf.argmax(logits), [1, -1])
    # pixel_label = tf.reshape(tf.argmax(ground_truth), [1, -1])

    Smax = tf.nn.softmax(logits, axis=3)

    softmax = Smax[:, :, :, 1]

    print('softmax: ', softmax.shape)

    predicts = tf.greater(softmax, 0.2)

    print('predicts: ', predicts.shape, ' threshold: ', threshold)

    labels = ground_truth if num_classes == 1 else ground_truth[:,:,:,1]
    labels = tf.greater(labels, 0)

    return predicts, softmax


def main(_):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)

    print(FLAGS.flag_values_dict())

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
    num = int(np.floor(num / FLAGS.batch_size)) * FLAGS.batch_size
    in_data = np.empty([num, im_height, im_width, 3], dtype=np.float32)
    for idx, number in enumerate(name_number):
        in_name = str(number) + "_sat.jpg"
        in_img = mpimg.imread(join(data_dir, in_name))
        in_data[idx, :, :, :] = in_img

    # Create graph
    in_pholder = tf.placeholder(tf.float32, in_data.shape)

    data_set = tf.data.Dataset.from_tensor_slices(in_pholder)
    data_set = data_set.batch(FLAGS.batch_size)

    data_iterator = data_set.make_initializable_iterator()
    next_element = data_iterator.get_next()

    image_pholder = tf.placeholder(tf.float32, [FLAGS.batch_size, 1024, 1024, 3])
    downSample = tf.image.resize_images(image_pholder, [FLAGS.side_length, FLAGS.side_length])

    prediction_down_img = tf.placeholder(in_data.dtype, [FLAGS.batch_size, FLAGS.side_length, FLAGS.side_length, 1])
    prediction_upSample = tf.image.resize_bilinear(prediction_down_img, [im_width, im_height])

    rgb_imgs, ground_truth, training_flag, logits, batch_loss = gripped_unet(in_channels = 3,
                                                                            out_channels = FLAGS.num_classes,
                                                                            num_filters = FLAGS.start_filters,
                                                                            side_length = FLAGS.side_length,
                                                                            num_convolutions = FLAGS.convolutions,
                                                                            kernel_size = FLAGS.kernel_size,
                                                                            depth = FLAGS.depth,
                                                                            batch_size = FLAGS.batch_size,
                                                                            data_format = FLAGS.data_format)

    print('ground_truth: ', ground_truth.shape, ', logits: ', logits.shape)

    threshold = 0.5
    im_predict, softmax = validate_graph(logits, ground_truth, FLAGS.num_classes, threshold)

    #im_predict = tf.cast(tf.greater(logits, 0), dtype=tf.float32)if FLAGS.num_classes == 1 else tf.argmax(logits, axis=3)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(data_iterator.initializer, feed_dict={in_pholder: in_data})
        sess.run(init_op)

        model_name = "%s/model_%d_%d_%d_%d.ckpt" % \
                    (FLAGS.model_dir, FLAGS.side_length, FLAGS.depth, FLAGS.start_filters, FLAGS.num_epochs)

        saver.restore(sess, model_name)

        prediction_result = np.empty([num, im_height, im_width, 1], dtype=np.uint8)

        for idx in range(int(num / FLAGS.batch_size)):
            image = sess.run(next_element) # (batch_size, 1024, 1024, 3)

            downsized_image = sess.run(downSample, feed_dict={image_pholder: image}) #(batch_size, 512, 512, 3)

            prediction = sess.run(im_predict, feed_dict={rgb_imgs: downsized_image, training_flag: False}) #(batch_size, 512,512, 1)

            prediction = prediction.reshape([FLAGS.batch_size, FLAGS.side_length, FLAGS.side_length, 1])

            prediction_1024 = sess.run(prediction_upSample, feed_dict={prediction_down_img: prediction}) #(batch_size,1024, 1024,1)

            prediction_result[idx, :, :, :] = prediction_1024.astype(np.uint8).reshape([im_height, im_width, 1])

        result_filename = "prediction_%s_%d" % (FLAGS.data, FLAGS.random_num)
        np.save(result_filename, prediction_result)


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('gpu_id', 3, 'id ranges from 0 to 3')
    tf.app.flags.DEFINE_integer('random_num', 3, 'random number')
    tf.app.flags.DEFINE_integer('side_length', 512, 'image')
    tf.app.flags.DEFINE_string('data', 'final_test', 'valid OR dev_test OR final_test')
    tf.app.flags.DEFINE_integer('num_epochs', 40, 'number of training epochs')
    tf.app.flags.DEFINE_integer('depth', 4, 'number of depth')
    tf.app.flags.DEFINE_integer('num_classes', 2, 'number of segmentation classes')
    tf.app.flags.DEFINE_integer('start_filters', 32, 'channels at full-res level')
    tf.app.flags.DEFINE_integer('convolutions', 4, 'trailing convolutional layers')
    tf.app.flags.DEFINE_integer('kernel_size', 3, 'convolution kernel size')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'number of images per mini-batch')
    tf.app.flags.DEFINE_string('data_format', 'channels_first', 'NCHW: channels_first, NHWC: channels_last')
    tf.app.flags.DEFINE_string('model_dir', '/home/rueisung/deep_globe/models/05042018', 'model directory')
    tf.app.flags.DEFINE_string('test_dir', '/media/workspace/bgong/data/road-extraction/', 'test data directory')
    tf.app.flags.DEFINE_string('valid_dir', '/media/workspace/bgong/data/road-extraction/train', 'validation data directory')
    tf.app.flags.DEFINE_string('name_dir', '/home/chenny1229/parameters/512/names/', 'name directory')
    # name_dir need to be changed to bgong (cp names to bgong directory)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

