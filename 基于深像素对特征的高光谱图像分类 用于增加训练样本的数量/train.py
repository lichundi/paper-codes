import scipy.io as sio
import tensorflow as tf
import numpy as np
import math


fea1 = 10
fea2 = fea1
fea3 = int(2*fea2)
fea4 = fea3
fea5 = int(2*fea4)
fea6 = fea5
fea7 = int(2*fea6)
fea8 = fea7

batchsize = 256

NUM_EPOCHES = 3
NUM_EPOCHS_PER_DECAY = 300
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999

sess = tf.InteractiveSession()#在运行图的时候，插入一些计算图


# import data from mat file
train_dataf = 'traindata.mat'
train_labelf = 'trainlabel.mat'



train_data = sio.loadmat(train_dataf)['traindata']

train_labels = sio.loadmat(train_labelf)['trainlabel']


classes = np.max(train_labels)   # max() 方法返回给定参数的最大值,
deepth = train_data.shape[3] # train_data.shape[3]第三维的长度


train_images = train_data.transpose((0,2,3,1))

train_labels -= 1

#tf.placeholder(dtype,shape=None,name=None) dtype 数据类型 shape 数据形状 name名称 
images_placeholder = tf.placeholder(tf.float32, shape=(None, 2, deepth, 1))
label_placeholder = tf.placeholder(tf.int64, shape=(None, 1))


class DataSet(object):

    def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape,
                                                     labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
        if dtype == tf.float32:
            images = images.astype(np.float32)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    @property  #Python内置的@property装饰器就是负责把一个方法变成属性调用的：
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


trainset = DataSet(train_images, train_labels, dtype=tf.float32)


def loss(logpros, labels):
    labels = tf.reshape(labels, [batchsize])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logpros, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op


def _variable_on_gpu(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd=0.0005):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_relu(input, kernel, bias, stride, padding):
    conv = tf.nn.conv2d(input, kernel, stride, padding=padding)
    return tf.nn.relu(conv+bias)


def inference(images):
    with tf.variable_scope('conv1') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,8,1,fea1], stddev=math.sqrt(1.0/8))
        biases = _variable_on_gpu('biases', [fea1], tf.constant_initializer(0.0))
        conv1 = conv_relu(images, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv1_') as scope:
        weights = _variable_with_weight_decay('weights', shape=[2,1,fea1,fea1], stddev=math.sqrt(1.0/fea1))
        biases = _variable_on_gpu('biases', [fea1], tf.constant_initializer(0.0))
        conv1_ = conv_relu(conv1, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea1,fea2], stddev=math.sqrt(2.0/3/fea1))
        biases = _variable_on_gpu('biases', [fea2], tf.constant_initializer(0.0))
        conv2 = conv_relu(conv1_, weights, biases, [1,1,1,1], 'SAME')
    pool1 = tf.nn.max_pool(conv2, [1,1,3,1], [1,1,3,1], padding='SAME')

    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea2,fea3], stddev=math.sqrt(2.0/3/fea2))
        biases = _variable_on_gpu('biases', [fea3], tf.constant_initializer(0.0))
        conv3 = conv_relu(pool1, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea3,fea4], stddev=math.sqrt(2.0/3/fea3))
        biases = _variable_on_gpu('biases', [fea4], tf.constant_initializer(0.0))
        conv4 = conv_relu(conv3, weights, biases, [1,1,1,1], 'VALID')
    pool2 = tf.nn.max_pool(conv4, [1,1,2,1], [1,1,2,1], padding='SAME')

    with tf.variable_scope('conv5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea4,fea5], stddev=math.sqrt(2.0/3/fea4))
        biases = _variable_on_gpu('biases', [fea5], tf.constant_initializer(0.0))
        conv5 = conv_relu(pool2, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('conv6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea5,fea6], stddev=math.sqrt(2.0/3/fea5))
        biases = _variable_on_gpu('biases', [fea6], tf.constant_initializer(0.0))
        conv6 = conv_relu(conv5, weights, biases, [1,1,1,1], 'VALID')
    pool3 = tf.nn.max_pool(conv6, [1,1,2,1], [1,1,2,1], padding='SAME')
    
    with tf.variable_scope('fc_conv1') as scope:
        dims = 5
        weights = _variable_with_weight_decay('weights', shape=[1,dims,fea6,fea7], stddev=math.sqrt(2.0/dims/fea6))
        biases = _variable_on_gpu('biases', [fea7], tf.constant_initializer(0.0))
        fc_conv1 = conv_relu(pool3, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('fc_conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea7,fea8], stddev=math.sqrt(2.0/fea7))
        biases = _variable_on_gpu('biases', [fea8], tf.constant_initializer(0.0))
        fc_conv2 = conv_relu(fc_conv1, weights, biases, [1,1,1,1], 'VALID')
    with tf.variable_scope('scores') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea8,classes], stddev=math.sqrt(1.0/fea8))
        biases = _variable_on_gpu('biases', [classes], tf.constant_initializer(0.0))
        scores = tf.nn.conv2d(fc_conv2, weights, [1,1,1,1], padding='VALID') + biases
    logits_flat = tf.reshape(scores, [-1, classes])

    return logits_flat


def trainop(total_loss, global_step):
    num_batches_per_epoch = trainset.num_examples / batchsize
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.scalar_summary('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train(reuse=None):
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('inference', reuse=reuse) as scope:
        logits = inference(images_placeholder)
    loss_ = loss(logits, label_placeholder)
    train_op = trainop(loss_, global_step)
    
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    
    sess.run(init)
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter('hyper_log', graph_def=graph_def)
    max_step = int((trainset.num_examples/batchsize)*NUM_EPOCHS_PER_DECAY*NUM_EPOCHES)
    for step in xrange(max_step):
        image, label = trainset.next_batch(batchsize)
        sess.run(train_op, feed_dict={images_placeholder: image, label_placeholder: label})
        if step % 100 == 0:
            summary_str,lv = sess.run([summary_op,loss_], feed_dict={images_placeholder: image, 
                label_placeholder: label})
            print('step %d, loss %f' % (step, lv))
            summary_writer.add_summary(summary_str, step)
        if step % 10000 == 0 or (step + 1) == max_step:
            saver.save(sess, 'checkpoint/model.ckpt', global_step=step)


train(None)
