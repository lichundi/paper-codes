
# coding: utf-8

# In[1]:

# do some setup
import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
import math
import time

# In[2]:

fea1 = 10
fea2 = fea1
fea3 = int(2*fea2)
fea4 = fea3
fea5 = int(2*fea4)
fea6 = fea5
fea7 = int(2*fea6)
fea8 = fea7
# fea1 = 16
# fea2 = fea1
# fea3 = 2*fea2
# fea4 = fea3
# fea5 = 2*fea4
# fea6 = fea5
# fea7 = 2*fea6
# fea8 = fea7

sess = tf.InteractiveSession()
window_size = '5'
#window_size = '5'
#window_size = '7'
test_dataf = 'testdata'+window_size+'.mat'
test_labelf = 'testlabel.mat'

test_label_widthf = 'testlabelwidth'+window_size+'.mat'

# test_data = sio.loadmat(test_dataf)['testdata']
test_data = np.array(h5py.File(test_dataf)['testdata']).T
test_labels = sio.loadmat(test_labelf)['testlabel']
test_label_width = sio.loadmat(test_label_widthf)['test_label_width']

classes = np.max(test_labels)+1
deepth = test_data.shape[3]
channel = test_data.shape[1]
# In[4]:
test_images = test_data.transpose((0,2,3,1))
test_labels -= 1

num_of_test = test_data.shape[0]
images_placeholder = tf.placeholder(tf.float32, shape=(None, 2, deepth, 1))
label_placeholder = tf.placeholder(tf.int64, shape=(None, 1))

# In[5]:

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
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            # images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
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

testset = DataSet(test_images, np.zeros(num_of_test), dtype=tf.float32)

def loss(logpros, labels):
    labels = tf.reshape(labels, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logpros, labels, name='xentropy')
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
    # drop1 = tf.nn.dropout(pool1, keep_prob)

    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea2,fea3], stddev=math.sqrt(2.0/3/fea2))
        biases = _variable_on_gpu('biases', [fea3], tf.constant_initializer(0.0))
        conv3 = conv_relu(pool1, weights, biases, [1,1,1,1], 'VALID')
    # drop_conv3 = tf.nn.dropout(conv3, keep_prob, noise_shape=[batchsize, 1, 1, fea3])
    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea3,fea4], stddev=math.sqrt(2.0/3/fea3))
        biases = _variable_on_gpu('biases', [fea4], tf.constant_initializer(0.0))
        conv4 = conv_relu(conv3, weights, biases, [1,1,1,1], 'VALID')
    pool2 = tf.nn.max_pool(conv4, [1,1,2,1], [1,1,2,1], padding='SAME')
    # drop_pool2 = tf.nn.dropout(pool2, keep_prob, noise_shape=[batchsize, 1, 1, fea4])

    with tf.variable_scope('conv5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea4,fea5], stddev=math.sqrt(2.0/3/fea4))
        biases = _variable_on_gpu('biases', [fea5], tf.constant_initializer(0.0))
        conv5 = conv_relu(pool2, weights, biases, [1,1,1,1], 'VALID')
    # drop_conv5 = tf.nn.dropout(conv5, keep_prob, noise_shape=[batchsize, 1, 1, fea5])
    with tf.variable_scope('conv6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,3,fea5,fea6], stddev=math.sqrt(2.0/3/fea5))
        biases = _variable_on_gpu('biases', [fea6], tf.constant_initializer(0.0))
        conv6 = conv_relu(conv5, weights, biases, [1,1,1,1], 'VALID')
    pool3 = tf.nn.max_pool(conv6, [1,1,2,1], [1,1,2,1], padding='SAME')
    # drop_pool3 = tf.nn.dropout(pool3, keep_prob, noise_shape=[batchsize, 1, 1, fea6])
    
    with tf.variable_scope('fc_conv1') as scope:
        dims = 5
        weights = _variable_with_weight_decay('weights', shape=[1,dims,fea6,fea7], stddev=math.sqrt(2.0/dims/fea6))
        biases = _variable_on_gpu('biases', [fea7], tf.constant_initializer(0.0))
        # scores = tf.nn.conv2d(pool3, weights, [1,1,1,1], padding='VALID')+biases
        fc_conv1 = conv_relu(pool3, weights, biases, [1,1,1,1], 'VALID')
        # fc_conv1_drop = tf.nn.dropout(fc_conv1, keep_prob)
    with tf.variable_scope('fc_conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea7,fea8], stddev=math.sqrt(2.0/fea7))
        biases = _variable_on_gpu('biases', [fea8], tf.constant_initializer(0.0))
        fc_conv2 = conv_relu(fc_conv1, weights, biases, [1,1,1,1], 'VALID')
    #     # fc_conv2_drop = tf.nn.dropout(fc_conv2, keep_prob)
    with tf.variable_scope('scores') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1,1,fea8,classes], stddev=math.sqrt(1.0/fea8))
        biases = _variable_on_gpu('biases', [classes], tf.constant_initializer(0.0))
        scores = tf.nn.conv2d(fc_conv2, weights, [1,1,1,1], padding='VALID') + biases
    logits_flat = tf.reshape(scores, [-1, classes])

    return logits_flat
    
# load from check point
def load_ckpt():
    # global_step = tf.Variable(0, trainable=False)
    # saver = tf.train.Saver(tf.all_variables())
    with tf.variable_scope('inference') as scope:
        logits = inference(images_placeholder)
    loss_ = loss(logits, label_placeholder)
    
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state('checkpoint/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found!')
        return


# In[21]:

load_ckpt()


# In[22]:

def test():
    images = testset.images
    prediction_list = []
    step = 2000
    index = 0
    with tf.variable_scope('inference', reuse=True) as scope:
        logpros = inference(images_placeholder)
        y_conv = tf.nn.softmax(logpros)
        y_ = tf.slice(y_conv, [0,1], [-1,classes-1])
    while index < num_of_test:
    	# print index
    	# t1 = time.time()
        if index + step > num_of_test:
            input_image = images[index:, :, :, :]
        else:
            input_image = images[index:(index+step), :, :, :]
        index += step
        
        # predictions = sess.run(tf.argmax(y_conv, 1))
        predictions = sess.run(tf.argmax(y_, 1), feed_dict={images_placeholder: input_image})
        
        prediction_list.extend(predictions)
        # print time.time() - t1
    return prediction_list

f = open('prediction.txt', 'w')
start=time.time()
pre_list = test()
print time.time()-start
pre_index = 0
matrix = np.zeros((classes-1, classes-1))
n = test_labels.shape[0]
for i in xrange(n):
    predictions = pre_list[pre_index:pre_index+test_label_width[i][0]]
    pre_index += test_label_width[i][0]
    
    # pre_label = np.argmax(np.max(predictions, axis=0))
    pre_label = np.argmax(np.bincount(predictions))
    f.write(str(pre_label)+'\n')
    matrix[pre_label, test_labels[i, 0]] += 1
print np.int_(matrix)
print np.sum(np.trace(matrix))
print np.sum(np.trace(matrix)) / float(n)
