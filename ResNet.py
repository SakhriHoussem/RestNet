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
    return tf.Variable(tf.zeros([shape]),name='bias')

def conv2d(x,W,strides=1,padding='SAME'):
    return tf.nn.conv2d(x,W,[1,strides,strides,1],padding='SAME')

def pool_max_2x2(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)

def pool_avg(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.avg_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)

def batchNorm_layer(): print("---BatchNorm---")

def scale_layer(): print("-----Scale-----")

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
    batchNorm_layer()
    scale_layer()
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
    #return tf.nn.softmax(tf.nn.relu(matmul_fc))

def my_model(data,num_classes=21,num_filters=64,filter_size=7,padding='SAME',strides=2):
    conv_1=conv_layer(data,num_filters,filter_size,strides)
    pool_1=pool_max_2x2(conv_1,ksize=3)
    b_1=block1(pool_1)
    b_2=block2(b_1)
    b_3=block3(b_2)
    b_4=block4(b_3)
    b_5=block5(b_4)
    return tf.nn.relu(fc_layer(b_5,num_classes))

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
softmax=tf.nn.softmax(logits)

cost=tf.reduce_mean(tf.abs(y - softmax))
train=tf.train.GradientDescentOptimizer(0.05).minimize(cost)

correct_prediction=tf.equal(tf.argmax(softmax),tf.argmax(y))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("we have ",weights," weights",bias," bias  ------------")

batch_size = 10
iteration =10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file=tf.summary.FileWriter("graph/",sess.graph)
    for _ in range(iteration):
        indice = np.random.permutation(n)
        for i in range(n):
            min_batch = indice[i*batch_size:(i+1)*batch_size]
            print(_,"-- Train : ",sess.run(train,feed_dict={x:data[min_batch],y:labels[min_batch]}))

    #print("\n\nLogits of model : \n\n",sess.run(logits,feed_dict={x:data[0],y:labels[0]}))
    #print("Softmax of model : \n\n",sess.run(softmax,feed_dict={x:data[0],y:labels[0]}))
    print("Cost : ",sess.run(cost,feed_dict={x:data,y:labels}))
    print("Accuracy  : ",sess.run(acc,feed_dict={x:data,y:labels}))

print("--- %s seconds ---" % (np.round(time.time() - start_time)))
sess.close()