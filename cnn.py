# -*- coding:utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf
import os
import time
from datetime import datetime
import cnn_input
import cv2

# params
MINIBATCH_SIZE = 64
LEARNING_RATE = 0.1
MAX_STEP = 20000
TRAIN = False
EVL = False
KAGGLE = True

# 在cpu上定义常量
def variable_on_cpu(name, shape, initializer = tf.constant_initializer(0.1)):
	with tf.device('/cpu:0'):
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
	return var

# 在cpu上定义变量
def variables(name, shape, stddev):
	dtype = tf.float32
	var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev = stddev, dtype = dtype))
	return var

# network
def cnn(images):
	with tf.variable_scope('conv1') as scope:
		# 5 * 5 window 64 kernals
		weights = variables('weights', [5, 5 ,3, 64], 5e-2)
		# conv1
		conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
		bias = variable_on_cpu('bias', [64])
		conv1 = tf.nn.bias_add(conv, bias)
		# relu
		conv1 = tf.nn.relu(conv1, name = scope.name)
		# tensorboard
		tf.histogram_summary(scope.name + '/activations', conv1)

	with tf.variable_scope('pooling1_and_lrn') as scope:
		# pooling1
		pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3 ,1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')
		# 局部响应归一化
		norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75, name = 'norm1')

	with tf.variable_scope('conv2') as scope:
		# 5 * 5 window 64 kernals
		weights = variables('weights', [5, 5 ,64, 64], 5e-2)
		# conv2
		conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
		bias = variable_on_cpu('bias', [64])
		conv2 = tf.nn.bias_add(conv, bias)
		# relu
		conv2 = tf.nn.relu(conv2, name = scope.name)
		# tensorboard
		tf.histogram_summary(scope.name + '/activations', conv2)

	with tf.variable_scope('pooling2_and_lrn') as scope:
		# 局部响应归一化
		norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75, name = 'norm2')
		# pooling2
		pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3 ,1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')

	with tf.variable_scope('local3') as scope:
		# full connect 1 layer
		reshape = tf.reshape(pool2, [MINIBATCH_SIZE,-1])
		connect_numbers = reshape.get_shape()[1].value
		weights = variables('weights', shape=[connect_numbers,384], stddev=0.004)
		bias = variable_on_cpu('bias', [384])
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + bias, name = scope.name)
		# or for this type
		# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#local3 = tf.add(tf.matmul(reshape, weights), bias)
		#local3 = tf.nn.relu(local3)
		# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#tensorboard
		tf.histogram_summary(scope.name + '/activations', local3)

	with tf.variable_scope('local4') as scope:
		# full connect 2 layer
		weights = variables('weights', shape=[384,192], stddev=0.004)
		bias = variable_on_cpu('bias', [192])
		local4 = tf.nn.relu(tf.matmul(local3, weights) + bias, name = scope.name)
		# tensorboard
		tf.histogram_summary(scope.name + '/activations', local4)

	with tf.variable_scope('local5') as scope:
		# full connect 3 layer
		weights = variables('weights', [192, 10], stddev=1/192.0)
		bias = variable_on_cpu('bias', [10])
		local5 = tf.add(tf.matmul(local4, weights), bias, name = scope.name)
		tf.histogram_summary(scope.name + '/activations', local5)

	return local5

# loss
def losses(pred, labels):
	with tf.variable_scope('loss') as scope:
		# change the dtype
		labels = tf.cast(labels, tf.int64)
		# softmax
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, labels, name='cross_entropy_per_example')
		loss = tf.reduce_mean(cross_entropy, name = 'loss')
		# tensprboard
		tf.scalar_summary(scope.name + '/x_entropy', loss)
	return loss

# train
def train():
	# global_step
	global_step = tf.Variable(0, name = 'global_step', trainable=False)
	# data files
	# TensorFlow源码默认下载文件夹
	data_dir = "/home/fuyan/kaggle/CIFAR_10/new_tensorflow/data/"
	# train_dir
	train_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/cifar10_train/'
	# load data
	images, labels = cnn_input.inputs(data_dir, MINIBATCH_SIZE)

	# graph
	# get loss
	loss = losses(cnn(images), labels)
	# use SGD without learning_rate_decay
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
	# BP
	bp = optimizer.minimize(loss, global_step = global_step)
	# save model
	saver = tf.train.Saver(tf.all_variables())
	# tensorboard
	summary_op = tf.merge_all_summaries()
	# init
	init = tf.initialize_all_variables()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
	config = tf.ConfigProto()
	# 占用 GPU 的 20% 资源
	config.gpu_options.per_process_gpu_memory_fraction = 0.2
	# 设置会话模式，用 InteractiveSession 可交互的会话，逼格高
	sess = tf.InteractiveSession(config=config)
	# 运行初始化
	sess.run(init)

	# 设置多线程协调器
	coord = tf.train.Coordinator()       
	# 开始 Queue Runners (队列运行器)
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	# 把汇总写进 train_dir，注意此处还没有运行
	summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

	# train start
	try:
		for step in xrange(MAX_STEP):
			if coord.should_stop():
				break
			start_time = time.time()
			# 在会话中运行 loss
			_, loss_value = sess.run([bp, loss])
			duration = time.time() - start_time
			# 确认收敛
			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'                
			if step % 30 == 0:
				# 本小节代码设置一些花哨的打印格式，可以不用管
				num_examples_per_step = MINIBATCH_SIZE
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)                    
				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

			if step % 100 == 0:
				# 运行汇总操作， 写入汇总
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)                

			if step % 1000 == 0 or (step + 1) == MAX_STEP:
				# 保存当前的模型和权重到 train_dir，global_step 为当前的迭代次数
				checkpoint_path = os.path.join(train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

	except Exception, e:
		coord.request_stop(e)
	finally:
		coord.request_stop()
		coord.join(threads)

	sess.close()

def evaluate():

	data_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/data/'
	train_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/cifar10_train/'
	images, labels = cnn_input.inputs(data_dir, MINIBATCH_SIZE, train = False)

	logits = cnn(images) 
	saver = tf.train.Saver(tf.all_variables())        

	os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.2
	sess = tf.InteractiveSession(config=config)
	coord = tf.train.Coordinator()       
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	
	# 加载模型参数
	print("Reading checkpoints...")
	ckpt = tf.train.get_checkpoint_state(train_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]         
		saver.restore(sess, os.path.join(train_dir, ckpt_name))
		print('Loading success, global_step is %s' % global_step)
        
        
	try:
 		# 对比分类结果，至于为什么用这个函数，后面详谈       
		top_k_op = tf.nn.in_top_k(logits, labels, 1)
		true_count = 0
		step = 0
		while step < 157:
			if coord.should_stop():
				break
			predictions = sess.run(top_k_op)
			true_count += np.sum(predictions)
			step += 1
            
		precision = true_count / 10000
		print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
	except tf.errors.OutOfRangeError:
		coord.request_stop()
	finally:
		coord.request_stop()
		coord.join(threads)
        
	sess.close()

def kaggle_test():
	# 这里注意！恢复模型的时候 输入数据要改成batchsize的格式
	train_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/cifar10_train/'
	images = tf.placeholder(tf.float32, [32, 32, 3])
	image = tf.image.per_image_whitening(images)
	image = tf.tile(image, [64, 1, 1])
	image = tf.reshape(image, [-1, 32, 32, 3])
	logits = cnn(image) 
	saver = tf.train.Saver(tf.all_variables())        

	os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	sess = tf.InteractiveSession(config=config)
	
	# 加载模型参数
	print("Reading checkpoints...")
	ckpt = tf.train.get_checkpoint_state(train_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]         
		saver.restore(sess, os.path.join(train_dir, ckpt_name))
		print('Loading success, global_step is %s' % global_step)

	file_test = file('test.log', 'w')
	for num in range(1, 300001):
		img = cv2.imread('../data/test/' + str(num) + '.png')
		predict = sess.run(logits, feed_dict = {images : img})
		y_index = (list(predict[0]).index(predict[0].max()))
		file_test.write(str(num) + " : " + str(y_index) + '\n')
		if num % 1000 == 0:
			print num
	file_test.close()

if __name__ == '__main__':
	if TRAIN:
		train()
	elif KAGGLE:
		kaggle_test()
	elif EVL:
		evaluate()
