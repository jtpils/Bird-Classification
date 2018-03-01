#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import random
import tensorflow as tf
from PIL import Image

#判断文件是否为有效（完整）的图片   
#会出现漏检的情况 
def IsValidImage(pathfile):  
    bValid = True  
    try:  
       Image.open(pathfile).verify()  
    except:  
       bValid = False  
    return bValid  
  
  
def is_valid_jpg(jpg_file):    
    """判断JPG文件是否完整  
    """    
    suffix = jpg_file.split('.')[-1].lower()
    if suffix == 'jpg' or suffix == 'jpeg':    
        with open(jpg_file, 'rb') as f:    
            f.seek(-2, 2)    
            return f.read() == '\xff\xd9' #判定jpg是否包含结束字段    
    else:    
        return True  

  
#利用PIL库进行jpeg格式判定，但有些没有结束字段的文件检测不出来  
def is_jpg(filename):  
    try:  
        i=Image.open(filename)  
        return i.format =='JPEG'  
    except IOError:  
        return False
        
#compress data if needed,the file path is like 'images/0/1.jpg'
def  compress(file_dir,max_size):
    for dirname in os.listdir(file_dir):
        child = os.path.join(file_dir,dirname) 
        for filename in os.listdir(child): 
            file_path  = os.path.join(child,filename)  
            try:
                with  open(file_path,'rb') as im:
                    image = Image.open(im)
                    image.thumbnail(max_size, Image.ANTIALIAS)
                    # width = image.size[0]  
                    # height = image.size[1]
                    # image = image.crop((0, 0, width - 70, height))
                    # file_path = file_path.replace('beta','thumb') //your origin file path  and new file path
                    # temp = file_path.split('\\')
                    # file_dir = temp[0]+os.path.sep+temp[1]+os.path.sep+temp[2] 
                    # if not os.path.isdir(file_dir):
                        # os.makedirs(file_dir)
                    image.save(file_path,'jpeg')
            except IOError as e:
                os.remove(file_path)  # directly remove damaged pic

#shuffle data and the result txt files is not necessary
def shuffle_data(file_dir):
    data ={}
    total_train_list = []
    total_val_list = []
    total_test_list =[]  
    for dirname in os.listdir(file_dir):
        keys=[] 
        child = os.path.join(file_dir,dirname) 
        files_list = os.listdir(child)
        if len(files_list) > 710:
            files_list =random.sample(files_list, 700)
        for filename in files_list: 
            file_path  = os.path.join(child,filename)  
            if  is_jpg(file_path)  and IsValidImage(file_path) and is_valid_jpg(file_path):
                data[file_path] = str(int(dirname))
                keys.append(file_path)
            else:
                print file_path+' is not valid jpg'
        length = len(keys)
        if length == 0:
            continue
        size_val_list = int(0.15*length)
        val_list = random.sample(keys, size_val_list)
        keys = filter(lambda x: x not in val_list, keys)
        size_test_list = int(0.2*len(keys))
        test_list = random.sample(keys, size_test_list)
        train_list = filter(lambda x: x not in test_list, keys)
        total_train_list.extend(train_list) 
        total_val_list.extend(val_list)
        total_test_list.extend(test_list)

    np.random.shuffle(total_train_list)  #shuffle in the origin list
    np.random.shuffle(total_val_list)
    np.random.shuffle(total_test_list)

    with open('train_data.txt','a+') as f:
        for train in total_train_list:
            f.writelines(train+' ' + data[train] + '\n')

    with open('validation_data.txt','a+') as f:
        for val in total_val_list:
            f.writelines(val+' ' + data[val] + '\n')

    with open('test_data.txt','a+') as f:
        for test in total_test_list:
            f.writelines(test+' ' + data[test] + '\n')
            
    with open('total_count.txt','a+') as f:
        f.writelines('train ' + str(len(total_train_list))+ '\n')
        f.writelines('val '  + str(len(total_val_list)) + '\n')
        f.writelines('test ' + str(len(total_test_list)) + '\n')

def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images_path,save_dir,out_file_prefix,num_shards):

    with open(images_path, 'r') as f:
        data = f.readlines()
        length = len(data)
        instances_per_shard = length / num_shards
        if length % num_shards != 0:
            num_shards += 1
        index = 0
        for shard in range(num_shards):
            output_filename = ('%s-%.5d-of-%.5d' % (out_file_prefix, shard, num_shards)) 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_file = os.path.join(save_dir, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)
            shard_data = data[index:index+instances_per_shard]
            for line in shard_data:
                image_split = line.split(' ')
                if len(image_split) > 2:
                    print line
                    continue
                image_path = image_split[0]
                label = int(image_split[1])
                #origin jpeg and not decoded
                with tf.gfile.FastGFile(image_path, 'rb') as f:
                    image_buffer = f.read()      
                example = tf.train.Example(features=tf.train.Features(feature={
                        'label': _int64_feature(label),
                        'img_encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
                writer.write(example.SerializeToString())
            index += instances_per_shard
        writer.close()

if __name__ == '__main__':
    #compress(images,(500,500))
    shuffle_data('images')
    #convert_to_tfrecord('train_data.txt','tfrecords','train',100)
    #convert_to_tfrecord('validation_data.txt','tfrecords','validataion',25)
    #convert_to_tfrecord('test_data.txt','tfrecords','test',25)   

