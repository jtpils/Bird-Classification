#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np

def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss

def evaluation(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct_prediction = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct_prediction, tf.float16)
        accuracy = tf.reduce_mean(correct)
        num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(scope+'/accuracy', accuracy)
        return accuracy,num_correct_preds

def optimize(loss, learning_rate,momentum):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
        train_op = optimizer.minimize(loss)
        return train_op

def save_last_layers_weights(sess,vgg):
    last_layers_weights = []
    for v in vgg.parameters:
        if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print('Printing Trainable Variables :', sess.run(v).shape)
            last_layers_weights.append(sess.run(v))
            np.savez('last_layers.npz',last_layers_weights)
    print("Last two layer weights saved")


def load_initial_weights(weight_files,sess,trainable):
    # weights_dict = np.load(weight_files[0], encoding = 'bytes') 在python3以上可能需要,vgg.npy,vgg.npz保存参数格式有区别
    weights_dict = np.load(weight_files[0]).item()
    vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv3_4','conv4_1','conv4_2','conv4_3','conv4_4','conv5_1','conv5_2','conv5_3','conv5_4']
    for op_name in vgg_layers:
        with tf.variable_scope(op_name, reuse = True):
            var = tf.get_variable('W', trainable = trainable)
            print('Adding weights to',var.name)
            # sess.run(var.assign(weights_dict[op_name+'_b']))   
            sess.run(var.assign(weights_dict[op_name][0]))
            var = tf.get_variable('b', trainable = trainable)
            print('Adding weights to',var.name)
            # sess.run(var.assign(weights_dict[op_name+'_W']))
            sess.run(var.assign(weights_dict[op_name][1]))
    if len(weight_files) > 1:
        last_layers_weights = np.load(weight_files[1])
        with tf.variable_scope('low_rank', reuse = True):
            var = tf.get_variable('W', trainable = True)
            print('Adding weights to',var.name)
            sess.run(var.assign(last_layers_weights['arr_0'][0]))
            var = tf.get_variable('b', trainable = True)
            print('Adding weights to',var.name)
            sess.run(var.assign(last_layers_weights['arr_0'][1]))
        with tf.variable_scope('fc', reuse = True):
            var = tf.get_variable('W', trainable = True)
            print('Adding weights to',var.name)
            sess.run(var.assign(last_layers_weights['arr_0'][2]))
            var = tf.get_variable('b', trainable = True)
            print('Adding weights to',var.name)
            sess.run(var.assign(last_layers_weights['arr_0'][3]))
