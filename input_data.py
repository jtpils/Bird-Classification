#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

def get_batch(data_dir,tf_file_prefix,image_W,image_H,batch_size,capacity,shuffle):
    tf_record_pattern = os.path.join(data_dir, '%s-*' % tf_file_prefix)
    files = tf.gfile.Glob(tf_record_pattern)
    filename_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    # 解析读取的样例。
    features = tf.parse_single_example(
    serialized_example,
    features={
        'label':tf.FixedLenFeature([],dtype=tf.int64),
        'img_encoded':tf.FixedLenFeature([],dtype=tf.string),
    })
    
    image = tf.image.decode_jpeg(features['img_encoded'], channels=3)
    label = tf.cast(features['label'],tf.int32)  
    tf.image.random_flip_left_right(image)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.reshape(image,[image_W,image_H,3])
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity,
                                                min_after_dequeue=capacity-300)
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity
                                                )
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

def get_total_count(file_path):
    count ={}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            count[line.split(' ')[0]] = int(line.split(' ')[1])
    return count['train'],count['val'],count['test']




