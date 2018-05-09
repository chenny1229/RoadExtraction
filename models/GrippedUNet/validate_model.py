
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt

from layer_funs import gripped_unet


def validate_graph(logits, ground_truth, num_classes, threshold):

    # pixel_predict = tf.reshape(tf.argmax(logits), [1, -1])
    # pixel_label = tf.reshape(tf.argmax(ground_truth), [1, -1])

    Smax = tf.nn.softmax(logits, axis = 3)

    softmax = Smax[:, :, :, 1]

    print('softmax: ', softmax.shape)

    predicts = tf.greater(softmax, threshold)
    # predicts = tf.greater(softmax, 0.2)

    print('predicts: ', predicts.shape, ' threshold: ', threshold)


    labels = ground_truth if num_classes == 1 else ground_truth[:,:,:,1]
    labels = tf.greater(labels, 0)

    img_isect = tf.cast(tf.logical_and(predicts, labels), dtype = tf.float32)
    img_union = tf.cast(tf.logical_or(predicts, labels), dtype = tf.float32)

    isect_array = tf.reduce_sum(img_isect, axis=[1, 2])
    union_array = tf.reduce_sum(img_union, axis=[1, 2])

    gt_array = tf.reduce_sum(tf.cast(labels, dtype=tf.float32), axis=[1, 2])

    det = tf.reduce_mean(tf.div(isect_array, gt_array))

    iou = tf.reduce_mean(tf.div(isect_array, union_array))

    return predicts, softmax, iou, det


def main(_):

    def tf_parser(record):
        features = tf.parse_single_example(
            record,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        annotation = tf.decode_raw(features['label'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image = tf.reshape(image, tf.stack([height, width, 3]))
        annotation = tf.reshape(annotation, tf.stack([height, width, 1]))

        # convert to [height, width, 2] for softmax
        if FLAGS.num_classes == 1:
            annotation = tf.greater(annotation, 0)
        else:
            annotation = tf.concat([tf.less_equal(annotation, 0), tf.greater(annotation, 0)], axis=2)

        return image, annotation

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)

    # force batch_size to be 1
    FLAGS.batch_size = 1

    print(FLAGS.flag_values_dict())

    num_valid = 934

    print("num_valid: ", num_valid)
    
    data_dir = '/media/workspace/rueisung/TFrecord/%d' % FLAGS.side_length
    
    filenames = ['%s/%s' % (data_dir, FLAGS.valid_file)]
    print('filenames:', filenames)
   
    valid_set = tf.data.TFRecordDataset(filenames)

    valid_set = valid_set.map(tf_parser)
    valid_set = valid_set.batch(FLAGS.batch_size)
    valid_set = valid_set.repeat(FLAGS.num_epochs)
    valid_iterator = valid_set.make_initializable_iterator()
    valid_next_element = valid_iterator.get_next()

    rgb_imgs, ground_truth, training_flag, logits, batch_loss = gripped_unet(in_channels = 3,
                                                                             out_channels = FLAGS.num_classes,
                                                                             num_filters = FLAGS.start_filters,
                                                                             side_length = FLAGS.side_length,
                                                                             num_convolutions = FLAGS.convolutions,
                                                                             kernel_size = FLAGS.kernel_size,
                                                                             depth = FLAGS.depth,
                                                                             batch_size = FLAGS.batch_size,
                                                                             data_format = FLAGS.data_format,
                                                                             gradient_only = FLAGS.gradient_only,
                                                                             pyramid_loss_ratio= FLAGS.pyramid_loss_ratio)

    print('ground_truth: ', ground_truth.shape, ', logits: ', logits.shape)

    threshold = 0.3137
    print("threshold: " + str(threshold))

    predict, softmax, iou, det = validate_graph(logits, ground_truth, FLAGS.num_classes, threshold)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    valid_runables = {"softmax" : softmax, "predict": predict, "iou": iou, "detection": det}

    saver = tf.train.Saver()
        
    is_plot_result = False
    img_height = FLAGS.side_length

    with tf.Session() as sess:

        # shall I group all initializer?        
        sess.run(valid_iterator.initializer)
        
        sess.run(init_op)

        model_name = "%s/model_%s_%d_%.1f_%d.ckpt" % (FLAGS.model_dir, FLAGS.gradient_only, FLAGS.start_filters, FLAGS.pyramid_loss_ratio, FLAGS.epoch_idx)

        #model_name = "./%s/model_%d_%d_%d_%d.ckpt" % \
                     #(FLAGS.model_dir, FLAGS.side_length, FLAGS.depth, FLAGS.start_filters, FLAGS.num_epochs)

        #model_name = "/media/workspace/chenny1229/experiment_models/model_augment:True_60.ckpt"

        print(model_name)

        saver.restore(sess, model_name)

        # test on validation set                       
        avg_iou = 0
        avg_det = 0

        valid_img = np.zeros((img_height, img_height * 2, 3))
        prob_img = np.zeros((img_height, img_height))

        for idx in range(num_valid):

            images, labels = sess.run(valid_next_element) # image: (2, 256, 256, 3)  labels:(2, 256, 256, 2)

            # print(images.shape, labels.shape)

            result = sess.run(valid_runables, feed_dict={rgb_imgs : images,
                                                         ground_truth : labels,
                                                         training_flag : False})

            valid_img[:, 0:img_height, :] = images[0, :, :, :].reshape([img_height, img_height, 3])
            valid_img[:, img_height : 2*img_height, 1] = 255 * labels[0, :, :, 1].reshape([img_height, img_height])
            valid_img[:, img_height : 2*img_height, 0] = 255 * \
                                                         result["predict"][0, :, :].reshape([img_height, img_height])

            prob_img = 255 * result["softmax"][0, :, :].reshape([img_height, img_height])

            plt.imsave("./%s/%04d_mask.png" % (FLAGS.image_dir, idx), valid_img.astype(np.uint8))

            plt.imsave("./%s/%04d_prob.png" % (FLAGS.image_dir, idx), prob_img.astype(np.uint8))

            avg_iou += result['iou']

            avg_det += result['detection']

            print(idx, ", iou: ", result['iou'], ', detection: ', result['detection'])

        # print epoch result                       
        print('iou: %f, , det: %f' % (avg_iou / num_valid, avg_det / num_valid))


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('gpu_id', 0, 'id ranges from 0 to 3')
    tf.app.flags.DEFINE_integer('depth', 4, 'depth')
    tf.app.flags.DEFINE_integer('intervals', 5, 'save model after every 5 epochs')
    tf.app.flags.DEFINE_integer('epoch_idx', 60, 'the epoch index to restore model')
    tf.app.flags.DEFINE_integer('side_length', 512, 'image')
    tf.app.flags.DEFINE_integer('num_epochs', 60, 'number of training epochs')
    tf.app.flags.DEFINE_integer('num_classes', 2, 'number of segmentation classes')
    tf.app.flags.DEFINE_integer('start_filters', 32, 'channels at full-res level')
    tf.app.flags.DEFINE_integer('convolutions', 1, 'trailing convolutional layers')
    tf.app.flags.DEFINE_integer('kernel_size', 3, 'convolution kernel size')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'number of images per mini-batch')
    tf.app.flags.DEFINE_string('data_format', 'channels_first', 'channels-first or channels-last')
    tf.app.flags.DEFINE_string('train_file', 'train_512.tfrecords', 'tfrecord used for training')
    tf.app.flags.DEFINE_string('valid_file', 'valid_512.tfrecords', 'tfrecord used for validation')
    tf.app.flags.DEFINE_string('model_dir', '/media/workspace/bgong/models/grippedUNet/augment_shift_addPixel', 'directory to load model file')
    tf.app.flags.DEFINE_string('image_dir', 'augment_shift_addPixel_0.2', 'directory to save image files')
    tf.app.flags.DEFINE_float('pyramid_loss_ratio', 0.5, 'use only gradient or add pixel values as features')
    tf.app.flags.DEFINE_bool('gradient_only', False, 'use only gradient or add pixel values as features')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
