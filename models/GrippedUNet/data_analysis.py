import PIL
import numpy as np
import os
import argparse
import shutil
import sys
import cv2
from os import listdir
from os.path import join
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from sklearn.metrics import confusion_matrix

MAP_P2L = '/media/workspace/rueisung/TFrecord/512/valid_name_number_512.npy'
LABEL = '/media/workspace/bgong/data/road-extraction/train'
PREDICTION = '/home/chenny1229/parameters/512/augment_shift_noPixel_0.2'

def main(unused_argv):

    map_p2l = np.load(MAP_P2L)
    pred_dir = PREDICTION
    label_dir = LABEL
    num = len(map_p2l)
    
    # --------- graph ----------- #
    p = array_ops.placeholder(dtypes.float32, shape=(512, 512))
    l = array_ops.placeholder(dtypes.float32, shape=(512, 512))
    l = tf.greater(l, 127)

    metric_map = {}
    for threshold in range(30, 101, 10):
        p_temp = tf.greater(p, threshold)

        img_isect = tf.cast(tf.logical_and(p_temp, l), dtype=tf.float32)
        img_union = tf.cast(tf.logical_or(p_temp, l), dtype=tf.float32)

        isect_array = tf.reduce_sum(img_isect)
        union_array = tf.reduce_sum(img_union)
        gt_array = tf.reduce_sum(tf.cast(l, dtype=tf.float32))
        det_array = tf.reduce_sum(tf.cast(p_temp, dtype=tf.float32))

        metric_map['recall_' + str(threshold)] = tf.reduce_mean(tf.div(isect_array, gt_array))
        metric_map['precision_' + str(threshold)] = tf.reduce_mean(tf.div(isect_array, det_array))
        metric_map['miou_' + str(threshold)] = tf.reduce_mean(tf.div(isect_array, union_array))


    with tf.Session() as sess:
        sum_metric_map = {}
        for threshold in range(30, 101, 10):
            sum_metric_map['miou_' + str(threshold)] = 0
            sum_metric_map['precision_' + str(threshold)] = 0
            sum_metric_map['recall_' + str(threshold)] = 0

        for i in range(num):
            print(i)
            prediction = cv2.imread(join(pred_dir, '%04d_prob.png'%i))
            b, label, r = cv2.split(cv2.imread(join(pred_dir, '%04d_mask.png' %i)))
            label = label[:, 512:]
            prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)

            result = sess.run(metric_map, feed_dict={p: prediction, l: label})
            for threshold in range(30, 101, 10):
                sum_metric_map['miou_' + str(threshold)] += result['miou_' + str(threshold)]
                sum_metric_map['precision_' + str(threshold)] += result['precision_' + str(threshold)]
                sum_metric_map['recall_' + str(threshold)] += result['recall_' + str(threshold)]

        for threshold in range(30, 101, 10):
            sum_metric_map['miou_' + str(threshold)] /= num
            sum_metric_map['precision_' + str(threshold)] /= num
            sum_metric_map['recall_' + str(threshold)] /= num

        print('sum up...')
        for threshold in range(30, 101, 10):
            print(threshold, ": miou", sum_metric_map['miou_' + str(threshold)], "precision",
                  sum_metric_map['precision_' + str(threshold)], "recall", sum_metric_map['recall_' + str(threshold)])
        np.save('miou_recall_precision', sum_metric_map)


if __name__ == '__main__':
    tf.app.run()
