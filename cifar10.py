# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
from datetime import datetime

import numpy as np
from six.moves import xrange
import tensorflow as tf

import my_cifar10_input

BATCH_SIZE = 64
LEARNING_RATE = 0.1
MAX_STEP = 50000
TRAIN = False


# 用 get_variable 在 CPU 上定义常量
def variable_on_cpu(name, shape, initializer = tf.constant_initializer(0.1)):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, 
                              dtype = dtype)
    return var

 # 用 get_variable 在 CPU 上定义变量
def variables(name, shape, stddev): 
    dtype = tf.float32
    var = variable_on_cpu(name, shape, 
                          tf.truncated_normal_initializer(stddev = stddev, 
                                                          dtype = dtype))
    return var
    
# 定义网络结构
def inference(images):
    with tf.variable_scope('conv1') as scope:
        # 用 5*5 的卷积核，64 个 Feature maps
        weights = variables('weights', [5,5,3,64], 5e-2)
        # 卷积，步长为 1*1
        conv = tf.nn.conv2d(images, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # 通过 ReLu 激活函数
        conv1 = tf.nn.relu(bias, name = scope.name)
        # 柱状图总结 conv1
        tf.histogram_summary(scope.name + '/activations', conv1)
    with tf.variable_scope('pooling1_lrn') as scope:
        # 最大池化，3*3 的卷积核，2*2 的卷积
        pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')
        # 局部响应归一化
        norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')

    with tf.variable_scope('conv2') as scope:
        weights = variables('weights', [5,5,64,64], 5e-2)
        conv = tf.nn.conv2d(norm1, weights, [1,1,1,1], padding = 'SAME')
        biases = variable_on_cpu('biases', [64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope.name)
        tf.histogram_summary(scope.name + '/activations', conv2)
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9.0, 
                          beta = 0.75, name = 'norm1')        
        pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1],
                               padding = 'SAME', name='pool1')

    with tf.variable_scope('local3') as scope:
        # 第一层全连接
        reshape = tf.reshape(pool2, [BATCH_SIZE,-1])
        dim = reshape.get_shape()[1].value
        weights = variables('weights', shape=[dim,384], stddev=0.004)
        biases = variable_on_cpu('biases', [384])
        # ReLu 激活函数
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, 
                            name = scope.name)
        # 柱状图总结 local3
        tf.histogram_summary(scope.name + '/activations', local3)
        
    with tf.variable_scope('local4') as scope:
        # 第二层全连接
        weights = variables('weights', shape=[384,192], stddev=0.004)
        biases = variable_on_cpu('biases', [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, 
                            name = scope.name)
        tf.histogram_summary(scope.name + '/activations', local4)
        
    with tf.variable_scope('softmax_linear') as scope:
        # softmax 层，实际上不是严格的 softmax ，真正的 softmax 在损失层
        weights = variables('weights', [192, 10], stddev=1/192.0)
        biases = variable_on_cpu('biases', [10])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, 
                                name = scope.name)
        tf.histogram_summary(scope.name + '/activations', softmax_linear)
        
    return softmax_linear
# 交叉熵损失层             
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        # 交叉熵损失，至于为什么是这个函数，后面会说明。
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                            (logits, labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.scalar_summary(scope.name + '/x_entropy', loss)
    
    return loss
def train():
    # global_step
    global_step = tf.Variable(0, name = 'global_step', trainable=False)
    # cifar10 数据文件夹
    data_dir = '/tmp/cifar10_data/cifar-10-batches-bin/'
    # 训练时的日志logs文件，没有这个目录要先建一个
    train_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/cifar10_train/'
    # 加载 images，labels
    images, labels = my_cifar10_input.inputs(data_dir, BATCH_SIZE)

    # 求 loss
    loss = losses(inference(images), labels)
    # 设置优化算法，这里用 SGD 随机梯度下降法，恒定学习率
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    # global_step 用来设置初始化
    train_op = optimizer.minimize(loss, global_step = global_step)
    # 保存操作
    saver = tf.train.Saver(tf.all_variables())
    # 汇总操作
    summary_op = tf.merge_all_summaries()
    # 初始化方式是初始化所有变量
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

    # 开始训练过程
    try:        
        for step in xrange(MAX_STEP):
            if coord.should_stop():
                break
            start_time = time.time()
            # 在会话中运行 loss
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            # 确认收敛
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'                
            if step % 30 == 0:
                # 本小节代码设置一些花哨的打印格式，可以不用管
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)                    
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, 
                                     examples_per_sec, sec_per_batch))
            
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

    data_dir = '/tmp/cifar10_data/cifar-10-batches-bin/'
    train_dir = '/home/fuyan/kaggle/CIFAR_10/new_tensorflow/cifar10_train/'
    images, labels = my_cifar10_input.inputs(data_dir, BATCH_SIZE, train = False)

    logits = inference(images) 
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
        
if __name__ == '__main__':
    
    if TRAIN:
        train ()
    else:
        evaluate()
