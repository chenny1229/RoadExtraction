
# coding: utf-8

from __future__ import absolute_import, division, print_function
import numpy as np
from os import listdir
from os.path import isfile, join

#import matplotlib as mpl
# mpl.use('TkAgg')

#import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import tensorflow as tf


def shuffle_and_separate(name_number, in_data, out_data, percentage):
    # shuffle before separated
    new_images = np.copy(in_data)
    new_masks = np.copy(out_data)
    new_name_number = np.copy(name_number)
    permutation = np.random.permutation(len(in_data))
    for old_index, new_index in enumerate(permutation):
        new_name_number[new_index] = name_number[old_index]
        new_images[new_index] = in_data[old_index]
        new_masks[new_index] = out_data[old_index]
    
    train_in_data = new_images[0: int(len(in_data) * percentage)]
    train_out_data = new_masks[0: int(len(out_data) * percentage)]
    train_name_number = new_name_number[0: int(len(out_data) * percentage)]
    valid_in_data = new_images[int(len(in_data) * percentage):]
    valid_out_data = new_masks[int(len(out_data) * percentage):]
    valid_name_number = new_name_number[int(len(out_data) * percentage):]

    return train_in_data, valid_in_data, train_out_data, valid_out_data, train_name_number, valid_name_number


def write_TFrecord(in_data, out_data, tfrecords_filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    
    num_img, im_height, im_width = in_data.shape[0], in_data.shape[1], in_data.shape[2]
    
    for idx in range(num_img):
        in_img = in_data[idx]
        out_img = out_data[idx]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(im_height),
            'width': _int64_feature(im_width),
            'image': _bytes_feature(tf.compat.as_bytes(in_img.tostring())),
            'label': _bytes_feature(tf.compat.as_bytes(out_img.tostring()))}))

        writer.write(example.SerializeToString())
    
    writer.close()



def smooth_and_resize(img, depth, gauss, padding, data_format):

    dformat = "NCHW" if data_format == "channels_first" else "NHWC"

    pyramid = [ img ]
    for _ in range(depth):
        smooth = tf.nn.depthwise_conv2d(input = pyramid[-1],
                                        filter = gauss,
                                        strides = [1, 1, 1, 1],
                                        padding = padding,
                                        data_format = dformat)

        if data_format == "channels_first":
            pyramid.append(smooth[:, :, ::2, ::2])
        else:
            pyramid.append(smooth[:, ::2, ::2, :])

    return pyramid[-1]


def gripped_unet(in_channels=3,
                out_channels=1,
                side_length=1024,
                batch_size=1,
                data_format="channels_last"):

    padding = "SAME"

    # Define inputs and helper functions #
    with tf.variable_scope('place_holders'):

        rgb_imgs = tf.placeholder(tf.float32,
                                  shape=(batch_size, side_length, side_length, in_channels),
                                  name='rgb_imgs')

        ground_truth = tf.placeholder(tf.float32,
                                      shape=(batch_size, side_length, side_length, out_channels),
                                      name='ground_truth')
    
    if data_format == "channels_first":
        ground_truth = tf.transpose(ground_truth, perm=[0, 3, 1, 2])

    # constant kernels
    gauss = tf.constant([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 16

    gauss3 = tf.concat([gauss, gauss, gauss], axis=2)

    downsample_image = smooth_and_resize(rgb_imgs, 1, gauss3, padding, data_format)
    downsample_label = smooth_and_resize(ground_truth, 1, gauss, padding, data_format)

    return downsample_image, downsample_label, rgb_imgs, ground_truth


def main():

    #img_dir = "/media/workspace/bgong/data/road-extraction/train"
    img_dir = "/Users/qichen/Documents/road_extraction/train1024_subset"

    in_names = [ name for name in listdir(img_dir) if name.endswith("_sat.jpg")]
    down_image_size = 512
    in_data = np.empty([len(in_names), down_image_size, down_image_size, 3], dtype = np.uint8)
    out_data = np.empty([len(in_names), down_image_size, down_image_size, 1], dtype = np.uint8)
    
    downsample_image, downsample_label, rgb_imgs, ground_truth = gripped_unet(in_channels=3, 
                                                                              out_channels=1, side_length=1024, 
                                                    batch_size=1, data_format="channels_last")
    
    name_number = []
    with tf.Session() as sess:
        for idx, in_name in enumerate(in_names):
            in_img = mpimg.imread(join(img_dir, in_name)).reshape(1, 1024, 1024, 3)
            out_name = in_name.replace("_sat.jpg", "_mask.png")
            out_img = mpimg.imread(join(img_dir, out_name))[:, :, 0].reshape(1, 1024, 1024, 1)
            
            number = in_name.replace("_sat.jpg", "")
            name_number.append(number)
            
            # Downsample
            if down_image_size != 1024:
                in_img = sess.run(downsample_image, feed_dict={rgb_imgs: in_img}) # (1, 512, 512, 3)
                out_img = sess.run(downsample_label, feed_dict={ground_truth: out_img}) # (1, 512, 512, 1)
                print(in_img)
            
            in_data[idx, :, :, :] = in_img
            out_data[idx, :, :, :] = out_img
            
        # shuffle and separate data into train and validation
    train_in_data, valid_in_data, train_out_data, valid_out_data, train_name_number, valid_name_number= shuffle_and_separate(
        name_number, in_data, out_data, 0.85)
        
    train_tfrecords = 'train_512.tfrecords'
    write_TFrecord(train_in_data, train_out_data, train_tfrecords)
    valid_tfrecords = 'valid_512.tfrecords'
    write_TFrecord(valid_in_data, valid_out_data, valid_tfrecords)
    np.save('train_name_number_512.npy', train_name_number)
    np.save('valid_name_number_512.npy', valid_name_number)


if __name__ == '__main__':
    main()

