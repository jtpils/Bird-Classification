#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import argparse
import os
import time
import logging  
import logging.handlers  
import input_data
from bilinear_vgg import bilinear_vgg
import tools
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LOG_FILE = 'bird.log'  
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 10240*1024, backupCount = 1000) # 实例化
fmt = '%(asctime)s- %(message)s' 
formatter = logging.Formatter(fmt)   # 实例化formatter  
handler.setFormatter(formatter)      # 为handler添加formatter  
logger = logging.getLogger('bird')    # 获取名为tst的logger  
logger.addHandler(handler)           # 为logger添加handler  
logger.setLevel(logging.INFO) 


tf.app.flags.DEFINE_integer('batch_size', 40, 'batch_size')
tf.app.flags.DEFINE_float('learning_rate', 0.9, 'learning_rate')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_boolean('train_all', False, 'train all or last two layers')
tf.app.flags.DEFINE_integer('num_epoch', 10, 'num of epoch')

FLAGS = tf.app.flags.FLAGS
logs_train_dir = 'logs/images_train/'
logs_val_dir = 'logs/images_val/'
checkpoint ='checkpoint/'
def run_training():
    num_classes = 1329
    IMG_W = 448
    IMG_H = 448
    CAPACITY = 1000
    train_dir = 'tfrecords'
    BATCH_SIZE = FLAGS.batch_size
    train_all = FLAGS.train_all
    learning_rate=FLAGS.learning_rate
    momentum = FLAGS.momentum
    num_epoch = FLAGS.num_epoch
    logger.info('learning_rate '+str(learning_rate))
    logger.info('num_epoch '+str(num_epoch))
    total_train_count,total_val_count,total_test_count =input_data.get_total_count('total_count.txt')
    train_batch, train_label_batch = input_data.get_batch(train_dir,
                                                  'train',
                                                 IMG_W,
                                                  IMG_H,
                                                 BATCH_SIZE, 
                                                  CAPACITY,True)
    val_batch, val_label_batch = input_data.get_batch(train_dir,
                                                  'validataion',
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY,False)
    test_batch, test_label_batch = input_data.get_batch(train_dir,
                                                  'test',
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY,False)  


    imgs = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_W, IMG_H, 3])
    labels = tf.placeholder(tf.int32, [BATCH_SIZE])
    keep_pro = tf.placeholder(tf.float32)
    vgg = bilinear_vgg(imgs,num_classes,train_all,keep_pro)
    loss = tools.loss(vgg.logits,labels)
    accuracy,num_correct_preds = tools.evaluation(vgg.logits,labels)
    optimizer = tools.optimize(loss,learning_rate,momentum)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        
        sess.run(tf.global_variables_initializer())

        weight_files =['vgg19.npy']
        if train_all == True:
            weight_files.append('last_layers.npz')
        tools.load_initial_weights(weight_files,sess,train_all)

        saver = tf.train.Saver()
        '''
        logger.info("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
           global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
           saver.restore(sess, ckpt.model_checkpoint_path)
           logger.info('Loading success, global_step is ' +  global_step)
        else:
           print('No checkpoint file found') 
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)

        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph) 

        total_batch = total_train_count / BATCH_SIZE
        total_val_batch = total_val_count / BATCH_SIZE
        for epoch in range(0,num_epoch):      
            for i in range(total_batch): 
                try:
                    batch_xs,batch_ys = sess.run([train_batch, train_label_batch])#左右两边命名不要一样
                    _ = sess.run(optimizer, feed_dict={imgs: batch_xs, labels: batch_ys,keep_pro:0.7})
                    if i % 50 == 0:
                      train_loss,train_accuracy,summary_str = sess.run([loss,accuracy,summary_op], feed_dict={imgs: batch_xs, labels: batch_ys,keep_pro:0.7})
                      train_writer.add_summary(summary_str, epoch * total_batch + i)
                      logger.info("Epoch: "+str(epoch)+" Step: "+str(i)+" Loss: "+ str(train_loss))
                      logger.info("Training Accuracy --> "+str(train_accuracy))

                      batch_val_x,batch_val_y = sess.run([val_batch, val_label_batch])
                      val_loss,val_accuracy,val_summary_str = sess.run([loss,accuracy,summary_op], feed_dict={imgs: batch_val_x, labels: batch_val_y,keep_pro:1.0})
                      val_writer.add_summary(val_summary_str, epoch * total_batch + i)
                      logger.info("val Loss: "+ str(train_loss))
                      logger.info("val Accuracy --> "+str(train_accuracy))                     
                except tf.errors.OutOfRangeError:
                      logger.info('batch out of range')  
                        
                break
            checkpoint_path = os.path.join(checkpoint,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=epoch)
            if train_all == False:
                tools.save_last_layers_weights(sess,vgg)
           # correct_val_count = 0
           # val_loss_total = 0.0
            #for i in range(total_val_batch):
             #   try:
              #      batch_val_x,batch_val_y = sess.run([val_batch, val_label_batch])
               #     val_loss,preds = sess.run([loss,num_correct_preds], feed_dict={imgs: batch_val_x, labels: batch_val_y})
                #    val_loss_total += val_loss
                 #   correct_val_count+=preds
                    #val_writer.add_summary(summary_str, epoch * total_batch + i)
               # except tf.errors.OutOfRangeError:
                #    logger.info('val batch out of range')
             #   break
            #logger.info("------------")
            #logger.info("Epoch: "+str (epoch+1)+" correct_val_count, total_val_count "+ str(correct_val_count)+" , "+str( total_val_count))
            #logger.info("Epoch: "+str (epoch+1)+ " Step: "+ str(i)+" Loss: "+str( val_loss_total/total_val_batch))                
            #logger.info("Validation Data Accuracy --> "+str( 100.0*correct_val_count/(1.0*total_val_count)))
            #logger.info("------------")
            #break
        correct_test_count = 0
        total_test_batch = total_test_count/BATCH_SIZE
        for i in range(total_test_batch):
            try:
                batch_test_x,batch_test_y = sess.run([test_batch, test_label_batch])
                preds = sess.run(num_correct_preds, feed_dict = {imgs: batch_test_x, labels: batch_test_y,keep_pro:1.0})
                correct_test_count+=preds
            except tf.errors.OutOfRangeError:
                logger.info ('test batch out of range')
            break
        logger.info("correct_test_count, total_test_count "+str(correct_test_count)+" , "+ str(total_test_count))
        logger.info("Test Data Accuracy --> "+str( 100.0*correct_test_count/(1.0*total_test_count)))


        coord.request_stop() 
        coord.join(threads)
def main(_):
    run_training()

if __name__=='__main__':
    tf.app.run()

