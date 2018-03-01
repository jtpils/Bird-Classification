#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf


class bilinear_vgg:
    def __init__(self,imgs,num_classes,trainable,keep_pro):
        self.imgs = imgs
        self.num_classes =  num_classes  
        self.trainable = trainable 
        self.parameters = []   
        self.keep_pro = keep_pro
        self.build()

    def build(self):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.imgs-mean
        vgg_cov = self.conv(self.imgs)
        bilinear = self.bilinear(vgg_cov,512,200,'low_rank')
        self.logits = self.fc_layers(bilinear)

    def conv(self,x):
        conv1_1 = self.conv_layer(x, 3, 64 ,'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 64, 64 ,'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, 64, 128 ,'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 128, 128,'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, 128, 256,'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 256, 256,'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 256, 256,'conv3_3')
        conv3_4 = self.conv_layer(conv3_3, 256, 256,'conv3_4')
        pool3 = self.max_pool(conv3_4, 'pool3')

        conv4_1 = self.conv_layer(pool3, 256, 512,'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 512, 512,'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 512, 512,'conv4_3')
        conv4_4 = self.conv_layer(conv4_3, 512, 512,'conv4_4')
        pool4 = self.max_pool(conv4_4, 'pool4')

        conv5_1 = self.conv_layer(pool4, 512, 512,'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 512, 512,'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 512, 512,'conv5_3')
        conv5_4 = self.conv_layer(conv5_3, 512, 512,'conv5_4')
        return conv5_4

    def conv_layer(self, x, in_channels, out_channels, name):
        with tf.variable_scope(name) :
            kernel = tf.get_variable("W",[3, 3, in_channels, out_channels],initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.1), trainable=self.trainable)
            out = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]
            return tf.nn.relu(out)     
    
    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC', name = name)

    def bilinear(self,x, in_channels, out_channels, name):
        #low-rank 
        with tf.variable_scope(name):
            kernel = tf.get_variable("W",[3, 3, in_channels, out_channels],initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.1), trainable=True)
            out = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]
            self.low_rank = tf.nn.relu(out)  
        
        low_rank_shape = self.low_rank.get_shape()
        print('Shape of low_rank', low_rank_shape)
        if len(low_rank_shape) == 4:
           size = low_rank_shape[1].value * low_rank_shape[2].value
        print ('final size',size)
        self.phi_I = tf.einsum('ijkm,ijkn->imn',self.low_rank,self.low_rank)
        print('Shape of phi_I after einsum', self.phi_I.get_shape())
        
        self.phi_I = tf.reshape(self.phi_I,[-1,out_channels*out_channels])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())
        

        self.phi_I = tf.divide(self.phi_I,float(size))  
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I),tf.sqrt(tf.abs(self.phi_I)+1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        return tf.nn.dropout(tf.nn.l2_normalize(self.y_ssqrt, dim=1),self.keep_pro)


    def fc_layers(self,x):
        input = x.get_shape()[1].value
        print('size of fc input', input)
        with tf.variable_scope('fc') as scope:
            fcw = tf.get_variable('W', [input, self.num_classes], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fcb = tf.get_variable("b", [self.num_classes], initializer=tf.constant_initializer(0.1), trainable=True)
            self.parameters += [fcw, fcb]
            return tf.nn.bias_add(tf.matmul(x, fcw), fcb)

