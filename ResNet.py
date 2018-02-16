import tensorflow as tf
import numpy as np
import time
from dataSetGenerator import dataSetGenerator
import matplotlib.pyplot as plt
from colors import *

start_time=time.time()
weights=0
bias=0

def img_to_tensor(img):
    return tf.convert_to_tensor(np.asarray(img, np.float32), np.float32)
"""
def weight_generater(shape, n):
    return tf.get_variable("weight"+str(n), shape=shape, initializer=tf.contrib.layers.xavier_initializer())
"""
def weight_generater(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: # xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def bias_generater(shape):
    return tf.Variable(tf.zeros([shape]), name="Weight")

def conv2d(x, W, strides=1, padding='SAME'):
    return tf.nn.conv2d(x, W, [1, strides, strides, 1], padding='SAME')

def pool_max(x, ksize=2, strides=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)

def pool_avg(x, ksize=2, strides=2, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)

def batchNorm_layer(inputs, is_training, decay = 1e-5, epsilon = 1e-3):
    scale = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()))
    beta = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()))
    pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()), trainable=False)
    pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def conv_layer(data, num_filters, filter_size, strides=1, padding="SAME", use_bias=True, use_relu=True,is_training = True):
    """

    :param data:
    :param num_filters:
    :param filter_size:
    :param strides:
    :param padding:
    :param use_bias:
    :param use_relu:
    :return:
            input
              |
             \/
        +------------+
          conv
          num_filters
          filter_size
          padding
          stride
        +------------+
          BatchNorm
        +------------+
          Scale
        +------------+
          relu
        +------------+
              |
             \/
    """
    # input shape [W, H, channels]
    if len(data.get_shape().as_list())==3 :
        height, width, channels=data.get_shape().as_list()
    elif len(data.get_shape().as_list())==4 :
        n, height, width, channels=data.get_shape().as_list()
    print(FAIL+"--conv size--\n"+ENDC, height, width, channels)
    # generate weigh [kernal size, kernal size, channel, number of filters]

    global weights
    weights += 1
    w=weight_generater([filter_size, filter_size, channels, num_filters], xavier_params=(1, height*width*channels))
    # for each filter W has his  specific bias
    b=bias_generater(num_filters)
    # reshape the input picture
    data=tf.reshape(data, [-1, height, width, channels])
    conv=conv2d(data, w, strides, padding)
    global bias
    if use_bias:
        bias += 1
        conv = tf.add(conv, b)
    conv = tf.add(batchNorm_layer(conv,is_training), b)
    bias += 1
    if use_relu: return tf.nn.relu(conv)
    return conv

def stepConvConv(data, num_filters=64, filter_size=1, strides=1 ,is_training = True):
    """
    :param data:
    :param num_filters:
    :param filter_size:
    :param strides:
    :return:

            input
              |
             \/
        +------------+
          conv
          num_filters 64
          filter_size 1
          padding 0
          stride 1
          False
        +------------+
          BatchNorm
          True
        +------------+
          Scale
          True
        +------------+
          relu
        +------------+
              |
             \/
        +------------+
          conv
          num_filters 64
          filter_size 3
          padding 1
          stride 1
          False
        +------------+
          BatchNorm
          True
        +------------+
          Scale
          True
        +------------+
          relu
        +------------+
              |
             \/
          +------------+
          conv
          num_filters 64*4
          filter_size 1
          padding 0
          stride 1
          False
        +------------+
          BatchNorm
        +------------+
          Scale
          True
        +------------+
              |
             \/
    """
    conv_1 = conv_layer(data, num_filters, filter_size, strides, padding="VALID", use_bias=False,is_training=is_training)
    conv_2 = conv_layer(conv_1, num_filters, 3, strides, padding='SAME', use_bias=False,is_training=is_training)
    return conv_layer(conv_2, num_filters*4, filter_size, strides, padding="VALID", use_bias=False, use_relu=False,is_training=is_training)

def stepConvConvConv(data, num_filters = 128, filter_size = 1, strides = 1,is_training=True):
    """

    :param data:
    :param num_filters:
    :param filter_size:
    :param strides:
    :return:

            input
              |
             \/
        +------------+
          conv
          num_filters 128
          filter_size 1
          padding 0
          stride 2
          False
        +------------+
          BatchNorm
          True
        +------------+
          Scale
          True
        +------------+
          relu
        +------------+
              |
             \/
        +------------+
          conv
          num_filters 128
          filter_size 3
          padding 1
          stride 1
          False
        +------------+
          BatchNorm
          True
        +------------+
          Scale
          True
        +------------+
          relu
        +------------+
              |
             \/
          +------------+
          conv
          num_filters 128*4
          filter_size 1
          padding 0
          stride 1
          False
        +------------+
          BatchNorm
        +------------+
          Scale
          True
        +------------+
              |
             \/
    """
    conv_1=conv_layer(data, num_filters, filter_size, 2, padding="VALID", use_bias=False,is_training=is_training)
    conv_2=conv_layer(conv_1, num_filters, 3, strides, padding='SAME', use_bias=False,is_training=is_training)
    return conv_layer(conv_2, num_filters*4, filter_size, strides, padding="VALID", use_bias=False, use_relu=False,is_training=is_training)

def ResidualBlock(data, num_filters=64,is_training=True):

    """
    :param data:
    :param num_filters:
    :return:
                          input
                            |
               ----------------------------
              |                           |
             \/                          \/
        +------------+              +------------+
          conv                       conv
          num_filters 128            num_filters 128*4
          filter_size 1              filter_size 1
          padding 0                  padding 0
          stride 2                   stride 1
          False                      False
        +------------+              +------------+
          BatchNorm                  BatchNorm
          True                       True
        +------------+              +------------+
          Scale                      Scale
          True                       True
        +------------+              +------------+
          relu                       relu
        +------------+              +------------+
              |                           |
             \/                           |
        +------------+                    |
          conv                            |
          num_filters 128                 |
          filter_size 3                   |
          padding 1                       |
          stride 1                        |
          False                           |
        +------------+                    |
          BatchNorm                       |
          True                            |
        +------------+                    |
          Scale                           |
          True                            |
        +------------+                    |
          relu                            |
        +------------+                    |
              |                           |
             \/                           |
          +------------+                  |
          conv                            |
          num_filters 128*4               |
          filter_size 1                   |
          padding 0                       |
          stride 1                        |
          False                           |
        +------------+                    |
          BatchNorm                       |
          True                            |
        +------------+                    |
          Scale                           |
          True                            |
        +------------+                    |
          relu                            |
        +------------+                    |
              |                           |
              ----------------------------
                           |
                          \/
                    +------------+
                         res
                    +------------+
                        relu
                    +------------+
                          |
                         \/
    """
    conv_1=conv_layer(data, num_filters*4, 1, 2, padding="VALID", use_bias=False, use_relu=False,is_training=is_training)
    conv_2=stepConvConvConv(data, num_filters, filter_size=1, strides=1,is_training=is_training)
    return tf.nn.relu(tf.add(conv_1, conv_2))


def block1(data, num_filters=64,is_training=True):
    """

    :param data:
    :param num_filters:
    :return:
                          input
                            |
               ----------------------------
              |                           |
             \/                          \/
        +------------+              +------------+
          conv                       conv
          num_filters 128            num_filters 128*4
          filter_size 1              filter_size 1
          padding 0                  padding 0
          stride 1                   stride 1
          False                      False
        +------------+              +------------+
          BatchNorm                  BatchNorm
          True                       True
        +------------+              +------------+
          Scale                      Scale
          True                       True
        +------------+              +------------+
          relu                       relu
        +------------+              +------------+
              |                           |
             \/                           |
        +------------+                    |
          conv                            |
          num_filters 128                 |
          filter_size 3                   |
          padding 1                       |
          stride 1                        |
          False                           |
        +------------+                    |
          BatchNorm                       |
          True                            |
        +------------+                    |
          Scale                           |
          True                            |
        +------------+                    |
          relu                            |
        +------------+                    |
              |                           |
             \/                           |
          +------------+                  |
          conv                            |
          num_filters 128*4               |
          filter_size 1                   |
          padding 0                       |
          stride 1                        |
          False                           |
        +------------+                    |
          BatchNorm                       |
          True                            |
        +------------+                    |
          Scale                           |
          True                            |
        +------------+                    |
          relu                            |
        +------------+                    |
              |                           |
              ----------------------------
                           |
                          \/
                    +------------+
                         res
                    +------------+
                        relu
                    +------------+
                          |
                         \/
    """
    print(BOLD+"BLOCK1 -----------------------------------"+ENDC)
    conv_1=stepConvConv(data, num_filters,is_training=is_training)
    conv_2=conv_layer(data, num_filters*4, 1, 1, padding="VALID", use_bias=False, use_relu=False,is_training=is_training)
    return tf.add(conv_1, conv_2)

def block2(data, num_filters=64,is_training=True):
    print(BOLD+"BLOCK2 -----------------------------------"+ENDC)
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    return ResidualBlock(data, num_filters*2,is_training=is_training)

def block3(data, num_filters=128,is_training=True):
    print(BOLD+"BLOCK3 -----------------------------------"+ENDC)
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    return ResidualBlock(data, num_filters*2,is_training=is_training)
def block4(data, num_filters=256,is_training=True):
    print(BOLD+"BLOCK4 -----------------------------------"+ENDC)
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    return ResidualBlock(data, num_filters*2,is_training=is_training)
def block5(data, num_filters=512,is_training=True):
    print(BOLD+"BLOCK5 -----------------------------------"+ENDC)
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data, num_filters,is_training=is_training)
    data=tf.nn.relu(tf.add(conv, data))
    print(WARNING+"---pooling---"+ENDC)
    return pool_avg(data, 7, 1, 'VALID')

def fc_layer(data, num_classes=21):
    print(BOLD+"FC LAYER -----------------------------------"+ENDC)
    n, height, width, channels=data.get_shape().as_list()
    print(FAIL+"---fc size---\n"+ENDC, height, width, channels)
    data=tf.reshape(data, [-1, height*width*channels])
    # generate weigh [kernal size, kernal size, channel, number of filters]
    global weights
    weights +=1
    w=weight_generater([height*width*channels, num_classes], xavier_params=(1, height*width*channels))
    # for each filter W has his  specific bias
    b=bias_generater(num_classes)
    return tf.add(tf.matmul(data, w), b,name='logits')

def ResNet50(data, num_classes=21, num_filters=64, filter_size=7, padding='SAME', strides=2,is_training=True):
    conv_1 = conv_layer(data, num_filters, filter_size, strides,is_training=is_training)
    pool_1 = pool_max(conv_1, ksize=3)
    print(WARNING+"---pooling---"+ENDC)
    b_1 = block1(pool_1,is_training=is_training)
    b_2 = block2(b_1,is_training=is_training)
    b_3 = block3(b_2,is_training=is_training)
    b_4 = block4(b_3,is_training=is_training)
    b_5 = block5(b_4,is_training=is_training)
    return fc_layer(b_5, num_classes)

if __name__ == '__main__':

    # Generate datSets
    path = "C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/"
    # path="C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    data, labels, classes = dataSetGenerator(path)

    # get image height, width, channels
    batche_num, height, width, channels = data.shape
    print("-Input image-\n", height, width, channels)

    # number of classes
    num_classes = len(classes)

    x = tf.placeholder(tf.float32, [None, height, width, channels],name='t_picture')
    y = tf.placeholder(tf.float32, [None, num_classes],name='t_labels')

    # sess = tf.InteractiveSession()
    logits=ResNet50(x, num_classes)
    # logits = tf.nn.softmax(logits)
    tf.add_to_collection('logits_op',logits)

    # Define a loss function
    loss=tf.reduce_mean(tf.abs(y-logits),name='Loss')
    tf.add_to_collection('loss_op',loss)

    # loss = tf.nn.softmax_cross_entropy_with_logits_v2 (labels=y, logits=logits)
    # loss=tf.nn.l2_loss(logits - y)
    # loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logits), reduction_indices=1))

    # train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    train = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss,name='Train_op')
    tf.add_to_collection('train_op',train)

    # train = tf.train.AdadeltaOptimizer().minimize(loss)

    # correct_prediction=tf.equal(tf.argmax(softmax), tf.argmax(y))
    # acc=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("\nWe have", weights, "weights", bias, "bias  ------------")

    batch_size = 10
    epochs = 1
    errors = []

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        file = tf.summary.FileWriter("graph/", sess.graph)
        for _ in range(epochs):
            print("*******************  ", _, "  *******************")
            indice = np.random.permutation(batche_num)
            for i in range(batch_size-1):
                min_batch = indice[i*batch_size:(i+1)*batch_size]
                curr_loss, curr_train = sess.run([loss, train], {x: data[min_batch], y: labels[min_batch]})
                print(_, "-Iteration %d\nloss:\n%s" % (i, curr_loss))
                errors.append(curr_loss)
        tf.add_to_collection('errors',errors)
        saver.save(sess, "Save/dataSaved")
    print("--- model trained in %s seconds ---" % (np.round(time.time() - start_time)))
    plt.plot(errors, label="loss")
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    plt.show()
    sess.close()