import tensorflow as tf
import numpy as np
import time
from dataSetGenerator import datSetGenerator

start_time=time.time()
weights=0
bias=0

def img_to_tensor(img):
    return tf.convert_to_tensor(np.asarray(img,np.float32),np.float32)

def weight_generater(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05),name="Weight")

def bias_generater(shape):
    return tf.Variable(tf.zeros([shape]),name="Weight")

def conv2d(x,W,strides=1,padding='SAME'):
    return tf.nn.conv2d(x,W,[1,strides,strides,1],padding='SAME')

def pool_max(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)

def pool_avg(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.avg_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)

def batchNorm_layer(inputs, is_training, decay = 0.999,epsilon = 1e-3):

    scale = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()))
    beta = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()))
    pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()), trainable=False)
    pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, epsilon)

def conv_layer(data,num_filters,filter_size,strides=1,padding="SAME",use_bias=True,use_relu=True):
    # input shape [W,H,channels]
    if len(data.get_shape().as_list())==3 :
        height,width,channels=data.get_shape().as_list()
    elif len(data.get_shape().as_list())==4 :
        n,height,width,channels=data.get_shape().as_list()
    print("--conv size--\n",height,width,channels)
    # generate weigh [kernal size,kernal size,channel,number of filters]
    w=weight_generater([filter_size,filter_size,channels,num_filters])
    global weights
    weights += 1
    # for each filter W has his  specific bias
    b=bias_generater(num_filters)
    #reshape the input picture
    data=tf.reshape(data,[-1,height,width,channels])
    conv=conv2d(data,w,strides,padding)

    if use_bias:
        global bias
        bias +=1
        conv=tf.add(conv,b)
    conv = tf.add(batchNorm_layer(conv,True),b)
    if use_relu: return tf.nn.relu(conv)
    return conv

def stepConvConv(data,num_filters=64,filter_size=1,strides=1):
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
    conv_1=conv_layer(data,num_filters,filter_size,strides,padding="VALID",use_bias=False)
    conv_2=conv_layer(conv_1,num_filters,3,strides,padding='SAME',use_bias=False)
    return conv_layer(conv_2,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)

def stepConvConvConv(data,num_filters=128,filter_size=1,strides=1):
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
    conv_1=conv_layer(data,num_filters,filter_size,2,padding="VALID",use_bias=False)
    conv_2=conv_layer(conv_1,num_filters,3,strides,padding='SAME',use_bias=False)
    return conv_layer(conv_2,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)

def blockend(data,num_filters=64):

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
    conv_1=conv_layer(data,num_filters*4,1,2,padding="VALID",use_bias=False,use_relu=False)
    conv_2=stepConvConvConv(data,num_filters,filter_size=1,strides=1)
    return tf.nn.relu(tf.add(conv_1, conv_2))


def block1(data,num_filters=64):
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
    print("BLOCK1 -----------------------------------")
    conv_1=stepConvConv(data,num_filters)
    conv_2=conv_layer(data,num_filters*4,1,1,padding="VALID",use_bias=False,use_relu=False)
    return tf.add(conv_1,conv_2)

def block2(data,num_filters=64):
    print("BLOCK2 -----------------------------------")
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)

def block3(data,num_filters=128):
    print("BLOCK3 -----------------------------------")
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)
def block4(data,num_filters=256):
    print("BLOCK4 -----------------------------------")
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)
def block5(data,num_filters=512):
    print("BLOCK5 -----------------------------------")
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data=tf.nn.relu(tf.add(conv, data))
    return pool_avg(data,7,1,'VALID')

def fc_layer(data,num_classes=21):
    print("FC LAYER -----------------------------------")
    n,height,width,channels=data.get_shape().as_list()
    print("---fc size---\n",height,width,channels)
    data=tf.reshape(data,[-1,height*width*channels])
    # generate weigh [kernal size,kernal size,channel,number of filters]
    w=weight_generater([height*width*channels,num_classes])
    # for each filter W has his  specific bias
    b=bias_generater(num_classes)
    return tf.add(tf.matmul(data,w),b)

def my_model(data,num_classes=21,num_filters=64,filter_size=7,padding='SAME',strides=2):
    conv_1=conv_layer(data,num_filters,filter_size,strides)
    pool_1=pool_max(conv_1,ksize=3)
    b_1=block1(pool_1)
    b_2=block2(b_1)
    b_3=block3(b_2)
    b_4=block4(b_3)
    b_5=block5(b_4)
    return fc_layer(b_5,num_classes)



if __name__ == '__main__':

    # Generate datSets
    path = "C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/"

    #path="C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    data,labels,classes=datSetGenerator(path)


    # get image height,width,channels
    n,height,width,channels=data.shape
    print("--Input image--\n",height,width,channels)


    # number of classes
    num_classes=len(classes)

    x=tf.placeholder(tf.float32,[None,height,width,channels])
    y=tf.placeholder(tf.float32,[None,num_classes])

    #sess = tf.InteractiveSession()
    logits=my_model(x,num_classes)

    # Define a loss function
    #loss=tf.reduce_mean(tf.abs(y- logits))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2 (labels=y, logits=logits)

    #loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logits), reduction_indices=1))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    #correct_prediction=tf.equal(tf.argmax(softmax),tf.argmax(y))
    #acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print("we have ",weights," weights",bias," bias  ------------")

    batch_size = 5
    iteration = 4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        file=tf.summary.FileWriter("graph/",sess.graph)
        for _ in range(iteration):
            s = sess.run(loss,feed_dict={x:data,y:labels})
            print("loss : ",s.shape,s)
            indice = np.random.permutation(n)
            for i in range(n):
                min_batch = indice[i*batch_size:(i+1)*batch_size]
                curr_loss,_ = sess.run([loss, train], {x:data[min_batch], y:labels[min_batch]})
                print("Iteration %d \n  loss: \n %s " % (i, curr_loss))

    print("--- %s seconds ---" % (np.round(time.time() - start_time)))
