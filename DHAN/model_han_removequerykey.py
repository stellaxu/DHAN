import tensorflow as tf
import numpy as np

from Dice import dice

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):

    self.u = tf.placeholder(tf.int32, [None,]) # [B]
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B] #sequence length
    self.lr = tf.placeholder(tf.float64, [])

    
    hidden_units = 128

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    # print("user_emb_w",user_emb_w.get_shape().as_list())
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    # print("item_emb_w",item_emb_w.get_shape().as_list())
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    # print("item_b",item_b.get_shape().as_list())
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
    
    # print("cate_list",cate_list.get_shape().as_list())

    ic = tf.gather(cate_list, self.i)


    # print("ic",ic.get_shape().as_list())
    i_emb = tf.concat(values = [
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
    # print("i_emb",i_emb.get_shape().as_list())
    i_b = tf.gather(item_b, self.i)

    jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
    j_b = tf.gather(item_b, self.j)

    hc = tf.gather(cate_list, self.hist_i) #[B,T]
    


    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

    hist_i_cat,hist_i_raw=attention_item(i_emb,h_emb,self.sl,hc,cate_count) #[B,C,H]
    print('hist_i_cat is {}'.format(hist_i_cat))
    print('hist_i_raw is {}'.format(hist_i_raw))

    
    # hist_i_raw =attention(i_emb, h_emb, self.sl)
    hist_i =attention(i_emb, hist_i_cat, tf.ones_like(self.sl)*cate_count)
    # print("hist_i",hist_i.get_shape().as_list())
    #-- attention end ---
    
    hist_i = tf.layers.batch_normalization(inputs = hist_i)
    hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
    hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')


    hist_i_raw = tf.layers.batch_normalization(inputs = hist_i_raw)
    hist_i_raw = tf.reshape(hist_i_raw, [-1, hidden_units], name='hist_raw_bn')
    hist_i_raw = tf.layers.dense(hist_i_raw, hidden_units, name='hist_raw_fcn')

    u_emb_i = hist_i

    u_emb_i_raw = hist_i_raw

    hist_j_cat,hist_j_raw=attention_item(j_emb,h_emb,self.sl,hc,cate_count)
    
    #hist_j =attention(j_emb, h_emb, self.sl)
    hist_j=attention(j_emb,hist_j_cat,tf.ones_like(self.sl)*cate_count)
    #-- attention end ---
    
    hist_j = tf.layers.batch_normalization(inputs = hist_j)
    hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
    hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

    hist_j_raw = tf.layers.batch_normalization(inputs = hist_j_raw)
    hist_j_raw = tf.reshape(hist_j_raw, [-1, hidden_units], name='hist_raw_bn')
    hist_j_raw = tf.layers.dense(hist_j_raw, hidden_units, name='hist_raw_fcn', reuse=True)



    u_emb_j = hist_j

    u_emb_j_raw = hist_j_raw


    print (u_emb_i.get_shape().as_list())
    print (u_emb_j.get_shape().as_list())
    print (i_emb.get_shape().as_list())
    print (j_emb.get_shape().as_list())
    #-- fcn begin -------
    din_i = tf.concat([u_emb_i,u_emb_i_raw, i_emb], axis=-1)
    din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
    d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
    #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
    # d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
    # d_layer_1_i = dice(d_layer_1_i, name='dice_1')
    # d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
    d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
    #d_layer_2_i = dice(d_layer_2_i, name='dice_2')
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
    din_j = tf.concat([u_emb_j,u_emb_j_raw, j_emb], axis=-1)
    din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
    d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    # d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
    # d_layer_1_j = dice(d_layer_1_j, name='dice_1')
    d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    # d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
    # d_layer_2_j = dice(d_layer_2_j, name='dice_2')
    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
    x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
    self.logits = i_b + d_layer_3_i
    
    #-- fcn end -------

    
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_b + d_layer_3_i)
    self.score_j = tf.sigmoid(j_b + d_layer_3_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
    # print (self.p_and_n.get_shape().as_list())


    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        )

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss

  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    return u_auc, socre_p_and_n
  
  def test(self, sess, uij):
    return sess.run(self.logits_sub, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
  

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def attention_item(queries, keys, keys_length,cate_idx,cate_count):
  '''
    queries:     [B, H]
    keys:        [B, T, H] T keys
    keys_length: [B]
  '''
  B=queries.get_shape().as_list()[0]
  T=keys.get_shape().as_list()[1]
  H=queries.get_shape().as_list()[-1]

  queries_hidden_units = queries.get_shape().as_list()[-1]   #i.e. H
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  # print("queries",queries.get_shape().as_list()) 
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  # print("queries",queries.get_shape().as_list()) 
  # din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  din_all = tf.concat([queries, keys, queries-keys], axis=-1)
  # print("din_all",din_all.get_shape().as_list()) 
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_1_all",d_layer_1_all.get_shape().as_list()) 
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_2_all",d_layer_2_all.get_shape().as_list()) 
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
  # print("outputs",outputs.get_shape().as_list()) 
  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # print("outputs",outputs.get_shape().as_list()) 
  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]
  print('outputs is {}'.format(outputs))
  
  #final_out=tf.Variable(tf.zeros([queries.get_shape().as_list()[0],cate_count,queries.get_shape().as_list()[-1]]))  #[B,C,H]
  

  for i in range(cate_count):
    print('cate_idx is {}'.format(cate_idx))
    idx_org=tf.equal(cate_idx,i) #[B,T] Boolean matrix
    print('idx_org is {}'.format(idx_org))
    idx=tf.expand_dims(idx_org,1) #[B,1,T] Boolean matrix
    print('idx is {}'.format(idx))

    weights=tf.where(idx,outputs,tf.zeros_like(outputs)) # [B,1,T]
    #weights=tf.expand_dims(weights,1) # [B,1,T]
    print('weights is {}'.format(weights))
    idx_new=tf.expand_dims(idx_org,-1) #[B,T,1]
    idx_key=tf.tile(idx_new,multiples=[1,1,H])#[B,T,H]
    print('idx_key is {}'.format(idx_key))
    ks=tf.where(idx_key,keys,tf.zeros_like(keys))  #[B,T,H]
    print('ks is {}'.format(ks))
    final_out=tf.matmul(weights,ks)   #[B,1,H]
    if i==0:
      a=final_out
    else:
      a=tf.concat([a,final_out],axis=1)   #[B,C,H]
      

  
  outputs1 = tf.matmul(outputs, keys)  # [B, 1, H]



  #cate_idx_j=tf.nn.embedding_lookup(cate_list,j_b)
  # print("outputs",outputs.get_shape().as_list()) 
  # Weighted sum
  #outputs = tf.matmul(outputs, keys)  # [B, 1, H]
  # print("outputs",outputs.get_shape().as_list()) 
  return a,outputs1

def attention_item_1(queries, keys, keys_length,cate_idx,cate_count):
  '''
    queries:     [B, H]
    keys:        [B, T, H] T keys
    keys_length: [B]
  '''
  B=queries.get_shape().as_list()[0]
  T=keys.get_shape().as_list()[1]
  H=queries.get_shape().as_list()[-1]

  queries_hidden_units = queries.get_shape().as_list()[-1]   #i.e. H
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  # print("queries",queries.get_shape().as_list()) 
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  # print("queries",queries.get_shape().as_list()) 
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  # print("din_all",din_all.get_shape().as_list()) 
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_1_all",d_layer_1_all.get_shape().as_list()) 
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_2_all",d_layer_2_all.get_shape().as_list()) 
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
  # print("outputs",outputs.get_shape().as_list()) 
  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # print("outputs",outputs.get_shape().as_list()) 
  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]
  print('outputs is {}'.format(outputs))
  
  #final_out=tf.Variable(tf.zeros([queries.get_shape().as_list()[0],cate_count,queries.get_shape().as_list()[-1]]))  #[B,C,H]
  

  for i in range(cate_count):
    print('cate_idx is {}'.format(cate_idx))
    idx_org=tf.equal(cate_idx,i) #[B,T] Boolean matrix
    print('idx_org is {}'.format(idx_org))
    idx=tf.expand_dims(idx_org,1) #[B,1,T] Boolean matrix
    print('idx is {}'.format(idx))

    weights=tf.where(idx,outputs,tf.zeros_like(outputs)) # [B,1,T]
    #weights=tf.expand_dims(weights,1) # [B,1,T]
    print('weights is {}'.format(weights))
    idx_new=tf.expand_dims(idx_org,-1) #[B,T,1]
    idx_key=tf.tile(idx_new,multiples=[1,1,H])#[B,T,H]
    print('idx_key is {}'.format(idx_key))
    ks=tf.where(idx_key,keys,tf.zeros_like(keys))  #[B,T,H]
    print('ks is {}'.format(ks))
    final_out=tf.matmul(weights,ks)   #[B,1,H]
    if i==0:
      a=final_out
    else:
      a=tf.concat([a,final_out],axis=1)   #[B,C,H]
      

  




  #cate_idx_j=tf.nn.embedding_lookup(cate_list,j_b)
  # print("outputs",outputs.get_shape().as_list()) 
  # Weighted sum
  #outputs = tf.matmul(outputs, keys)  # [B, 1, H]
  # print("outputs",outputs.get_shape().as_list()) 
  return a



def attention(queries, keys, keys_length): #keys_length=156
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  # print("queries",queries.get_shape().as_list()) 
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  # print("queries",queries.get_shape().as_list()) 
  # din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  din_all = tf.concat([queries, keys, queries-keys], axis=-1)
  # din_all=tf.contrib.layers.layer_norm(din_all)
  # print("din_all",din_all.get_shape().as_list()) 
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att_1', reuse=tf.AUTO_REUSE)
  # print("d_layer_1_all",d_layer_1_all.get_shape().as_list()) 
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att_1', reuse=tf.AUTO_REUSE)
  # print("d_layer_2_all",d_layer_2_all.get_shape().as_list()) 
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att_1', reuse=tf.AUTO_REUSE)
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  # print("d_layer_3_all",d_layer_3_all.get_shape().as_list()) 
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
  # print("outputs",outputs.get_shape().as_list()) 
  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # print("outputs",outputs.get_shape().as_list()) 
  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]
  # print("outputs",outputs.get_shape().as_list()) 
  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]
  # print("outputs",outputs.get_shape().as_list()) 
  return outputs

def attention_multi_items(queries, keys, keys_length):
  '''
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H] 
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries_nums = queries.get_shape().as_list()[1]
  queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # shape : [B, N, T, H]
  max_len = tf.shape(keys)[1]
  keys = tf.tile(keys, [1, queries_nums, 1])
  keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # shape : [B, N, T, H]
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
  key_masks = tf.tile(key_masks, [1, queries_nums])
  key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
  outputs = tf.reshape(outputs, [-1, 1, max_len])
  keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
  #print outputs.get_shape().as_list()
  #print keys.get_sahpe().as_list()
  # Weighted sum
  outputs = tf.matmul(outputs, keys)
  outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
  print (outputs.get_shape().as_list())
  return outputs

