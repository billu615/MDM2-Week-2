# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build the ResNet model."""
    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.compat.v1.summary.merge_all()

  def _build_model(self):
    with tf.compat.v1.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]

    with tf.compat.v1.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.compat.v1.variable_scope(f'unit_1_{i}'):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.compat.v1.variable_scope(f'unit_2_{i}'):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.compat.v1.variable_scope(f'unit_3_{i}'):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.compat.v1.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.compat.v1.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.compat.v1.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()
      tf.compat.v1.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.compat.v1.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.compat.v1.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.compat.v1.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  def _stride_arr(self, stride):
    return [1, stride, stride, 1]

  def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
      with tf.compat.v1.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.compat.v1.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.compat.v1.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.compat.v1.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool2d(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
      with tf.compat.v1.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.compat.v1.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.compat.v1.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter // 4, stride)

    with tf.compat.v1.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter // 4, out_filter // 4, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter // 4, out_filter, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    return x

  def _batch_norm(self, name, x):
    with tf.compat.v1.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.compat.v1.get_variable('beta', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(0.0))
      gamma = tf.compat.v1.get_variable('gamma', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(1.0))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        moving_mean = tf.compat.v1.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(1.0), trainable=False)
        self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
      else:
        mean = tf.compat.v1.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
        variance = tf.compat.v1.get_variable('moving_variance', params_shape, tf.float32, initializer=tf.compat.v1.constant_initializer(1.0), trainable=False)
        tf.compat.v1.summary.histogram(mean.op.name, mean)
        tf.compat.v1.summary.histogram(variance.op.name, variance)

      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _decay(self):
    costs = []
    for var in tf.compat.v1.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.compat.v1.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.compat.v1.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32, initializer=tf.compat.v1.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.compat.v1.get_variable('DW', [x.get_shape()[1], out_dim], initializer=tf.compat.v1.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.compat.v1.get_variable('biases', [out_dim], initializer=tf.compat.v1.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
