#-*- coding:utf-8 -*-

# this file aim to loading images from bin

from __future__ import division

import tensorflow as tf
import os

# image parms
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

class cifar10_data(object):

	def __init__(self, filename_queue):
		# image parms
		self.height = IMAGE_HEIGHT
		self.width = IMAGE_WIDTH
		self.depth = IMAGE_CHANNELS
		self.label_bytes = 1
		# image size
		self.image_bytes = self.height * self.width * self.depth
		self.record_bytes = self.label_bytes + self.image_bytes
		self.label, self.image = self.load_cifar10(filename_queue)

	def load_cifar10(self, filename_queue):

		# tensorflow reader
		reader = tf.FixedLengthRecordReader(record_bytes = self.record_bytes)
		key, value = reader.read(filename_queue)
		record_bytes = tf.decode_raw(value, tf.uint8)
		# tf.slice(record_bytes, 起始位置， 长度)
		# 用tf.slice进行切片
		label = tf.cast(tf.slice(record_bytes, [0], [self.label_bytes]), tf.int32)
		image_raw = tf.slice(record_bytes, [self.label_bytes], [self.image_bytes])
		# loaded data format is 3 * 32 * 32
		image_raw = tf.reshape(image_raw, [self.depth, self.height, self.width])
		# turn to 32  * 32 * 3 because this is tensorflow format
		image = tf.transpose(image_raw, (1, 2, 0))
		image = tf.cast(image, tf.float32)
		return label, image
		
def inputs(data_dir, batch_size, train = True, name = 'input'):

	# 建议加上 tf.name_scope, 可以画出漂亮的流程图。
	with tf.name_scope(name):
		if train: 
		# 要读取的文件的名字
			#filenames = [os.path.join(data_dir,'data_batch_%d.bin' % ii) for ii in range(1,6)]
			filenames = [os.path.join(data_dir,'batch_%d.bin' % ii) for ii in range(1,6)]
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