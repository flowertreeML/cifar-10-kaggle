#-*- coding:utf-8 -*-

from __future__ import absolute_import        # 绝对导入
from __future__ import division                # 精确除法，/是精确除，//是取整除
from __future__ import print_function        # 打印函数

import os
import tensorflow as tf

# 建立一个 cifar10_data 的类， 输入文件名队列，输出 labels 和images
class cifar10_data(object):

    def __init__(self, filename_queue):        # 类初始化
        
        # 根据上一篇文章介绍的文件格式，定义初始化参数
        self.height = 32
        self.width = 32
        self.depth = 3
        # label 一个字节
        self.label_bytes = 1
        # 图像 32*32*3 = 3072 字节
        self.image_bytes = self.height * self.width * self.depth
        # 读取的固定字节长度为 3072 + 1 = 3073 
        self.record_bytes = self.label_bytes + self.image_bytes
        self.label, self.image = self.read_cifar10(filename_queue)
        
    def read_cifar10(self, filename_queue):

        # 读取固定长度文件
        reader = tf.FixedLengthRecordReader(record_bytes = self.record_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)
        # tf.slice(record_bytes, 起始位置， 长度)
        label = tf.cast(tf.slice(record_bytes, [0], [self.label_bytes]), tf.int32)
        # 从 label 起，切片 self.image_bytes = 3072 长度为图像
        image_raw = tf.slice(record_bytes, [self.label_bytes], [self.image_bytes])
        # 图片转化成 3*32*32
        image_raw = tf.reshape(image_raw, [self.depth, self.height, self.width])
        # 图片转化成 32*32*3
        image = tf.transpose(image_raw, (1,2,0))        
        image = tf.cast(image, tf.float32)
        return label, image

        
def inputs(data_dir, batch_size, train = True, name = 'input'):

    # 建议加上 tf.name_scope, 可以画出漂亮的流程图。
    with tf.name_scope(name):
        if train: 
            # 要读取的文件的名字
            filenames = [os.path.join(data_dir,'data_batch_%d.bin' % ii) 
                        for ii in range(1,6)]
            # 不存在该文件的时候报错
            for f in filenames:
                if not tf.gfile.Exists(f):
                    raise ValueError('Failed to find file: ' + f)
            # 用文件名生成文件名队列
            filename_queue = tf.train.string_input_producer(filenames)
            # 送入 cifar10_data 类中
            read_input = cifar10_data(filename_queue)
            images = read_input.image
            # 图像白化操作，由于网络结构简单，不加这句正确率很低。
            images = tf.image.per_image_whitening(images)
            labels = read_input.label
            # 生成 batch 队列，16 线程操作，容量 20192，min_after_dequeue 是
            # 离队操作后，队列中剩余的最少的元素，确保队列中一直有 min_after_dequeue
            # 以上元素，建议设置 capacity = min_after_dequeue + batch_size * 3
            num_preprocess_threads = 16
            image, label = tf.train.shuffle_batch(
                                    [images,labels], batch_size = batch_size, 
                                    num_threads = num_preprocess_threads, 
                                    min_after_dequeue = 20000, capacity = 20192)
        
            
            return image, tf.reshape(label, [batch_size])
            
        else:
            filenames = [os.path.join(data_dir,'test_batch.bin')]
            for f in filenames:
                if not tf.gfile.Exists(f):
                    raise ValueError('Failed to find file: ' + f)
                    
            filename_queue = tf.train.string_input_producer(filenames)
            read_input = cifar10_data(filename_queue)
            images = read_input.image
            images = tf.image.per_image_whitening(images)
            labels = read_input.label
            num_preprocess_threads = 16
            image, label = tf.train.shuffle_batch(
                                    [images,labels], batch_size = batch_size, 
                                    num_threads = num_preprocess_threads, 
                                    min_after_dequeue = 20000, capacity = 20192)
        
            
            return image, tf.reshape(label, [batch_size])
