# 
# Copyright (C) 2002-2019 Igor Sysoev
# Copyright (C) 2011,2019 Nginx, Inc.
# Copyright (C) 2010-2019 Alibaba Group Holding Limited
# Copyright (C) 2011-2013 Xiaozhe "chaoslawful" Wang
# Copyright (C) 2011-2013 Zhang "agentzh" Yichun
# Copyright (C) 2011-2013 Weibin Yao
# Copyright (C) 2012-2013 Sogou, Inc.
# Copyright (C) 2012-2013 NetEase, Inc.
# Copyright (C) 2014-2017 Intel, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#  
#   THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
#   OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#   SUCH DAMAGE.
#  

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
#from keras import backend as K

class QAAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(QAAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)

  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    new_h = (1. - att_score) * state + att_score * c
    return new_h, new_h

class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    input_size = query.get_shape().as_list()[-1]

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        tmp1 = tf.tensordot(facts, w1, axes=1)
        tmp2 = tf.tensordot(query, w2, axes=1)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas

def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]]) #  b,t 
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output


def first_attention(query, facts, attention_size, mask, cate_idx,cate_count,stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]



    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        item_weight=scores

    # Weighted sum
    if mode == 'SUM':
        eps=1e-10

        for i in range(cate_count):
            print('cate_idx is {}'.format(cate_idx))
            idx_org=tf.equal(cate_idx,i) #[B,T] Boolean matrix
            print('idx_org is {}'.format(idx_org))
            idx=tf.expand_dims(idx_org,1) #[B,1,T] Boolean matrix
            print('idx is {}'.format(idx))

            weights=tf.where(idx,scores,tf.zeros_like(scores)) #[B,1,T]

            weights_org=tf.where(idx,scores,tf.zeros_like(scores)) #[B,1,T]
            weights_org_red=tf.squeeze(weights_org,[1]) #[B,T]
            weights_org_red_max=tf.reduce_max(weights_org_red,1) #B 
            weights_org_red_max=tf.expand_dims(weights_org_red_max,-1) #[B,1]
            weights_sum=tf.reduce_sum(weights_org_red,1) #[B] ---------2
            weights_sum=tf.expand_dims(weights_sum,1) #[B,1]
            weights_sum=weights_sum+eps #[B,1]
            weights=tf.divide(weights_org_red,weights_sum) #[B,T]
            weights=tf.expand_dims(weights,1) #[B,1,T]
            wt=tf.squeeze(weights,axis=[1]) #[B,T]

            #weights=tf.expand_dims(weights,1) # [B,1,T]
            print('weights is {}'.format(weights))
            idx_new=tf.expand_dims(idx_org,-1) #[B,T,1]
            idx_key=tf.tile(idx_new,multiples=[1,1,facts.get_shape().as_list()[2]])#[B,T,H]
            print('idx_key is {}'.format(idx_key))
            ks=tf.where(idx_key,facts,tf.zeros_like(facts))  #[B,T,H]
            print('ks is {}'.format(ks))
            final_out=tf.matmul(weights,ks)   #[B,1,H]
            print('final_out is {}'.format(final_out))
            if i==0:
              # output=final_out
              a = final_out
              wt_item=weights_org_red_max
            else:
              a=tf.concat([a,final_out],axis=1)   #[B,C,H]
              wt_item=tf.concat([wt_item,weights_org_red_max],axis=1)

        output=a
        outputs1=tf.matmul(scores,facts) #[B,1,H]
        print('cate_count is {}, output is {}'.format(cate_count,output))
        #output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]]) #[B,T]
        output = facts * tf.expand_dims(scores, -1) #[B,T,H]*[B,T,1]
        output = tf.reshape(output, tf.shape(facts))
    return output,outputs1

def second_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='sf1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='sf2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='sf3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]

    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T] only this commited


    key_masks=tf.count_nonzero(facts,2) # [B, T]
    key_masks=tf.cast(key_masks,dtype=tf.bool) # [B, T], true false matrix
    key_masks = tf.expand_dims(key_masks, 1) #[B,1,T]


    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
        cat_weight=scores

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output,cat_weight



def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch[:, 0:i+1, :],
                                               ATTENTION_SIZE, mask[:, 0:i+1], softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch,
                                               ATTENTION_SIZE, mask, softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1_trans_shine' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, facts_size, activation=tf.nn.sigmoid, name='f1_shine_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, facts_size, activation=tf.nn.sigmoid, name='f2_shine_att' + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all, tf.shape(facts))
    output = d_layer_2_all
    return output

