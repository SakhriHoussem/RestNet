import tensorflow as tf
import numpy as np
import cv2
import time
from dataSetGenerator import datSetGenerator

start_time=time.time()

def img_to_tensor(img):
    return tf.convert_to_tensor(np.asarray(img,np.float32),np.float32)

def weight_generater(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05),name="Weight")

def bias_generater(shape):
    return tf.Variable(tf.truncated_normal(shape=[shape]),name='bias')

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
    if len(tf.shape(data).eval())==3 :
        height,width,channels=tf.shape(data).eval()
    elif len(tf.shape(data).eval())==4 :
        n,height,width,channels=tf.shape(data).eval()
    # generate weigh [kernal size,kernal size,channel,number of filters]
    w=weight_generater([filter_size,filter_size,channels,num_filters])
    # for each filter W has his  specific bias
    b=bias_generater(num_filters)
    #reshape the input picture
    data=tf.reshape(data,[-1,height,width,channels])
    conv=conv2d(data,w,strides,padding)
    if use_bias: conv=tf.add(conv,b)
    batchNorm_layer()
    scale_layer()
    if use_relu: return tf.nn.relu(conv)
    return conv

def stepConvConv(data,num_filters=64,filter_size=1,strides=1):
    conv_1=conv_layer(data,num_filters,filter_size,strides,padding="VALID",use_bias=False)
    conv_2=conv_layer(conv_1,num_filters,3,strides,padding='SAME',use_bias=False)
    return conv_layer(conv_2,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)

def stepConvConvConv(data,num_filters=128,filter_size=1,strides=1):
    conv_1=conv_layer(data,num_filters,filter_size,2,padding="VALID",use_bias=False)
    conv_2=conv_layer(conv_1,num_filters,3,strides,padding='SAME',use_bias=False)
    return conv_layer(conv_2,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)

def block1(data,num_filters=64):
    print("BLOCK1 #########################")
    conv_1=stepConvConv(data,num_filters)
    conv_2=conv_layer(data,num_filters*4,1,1,padding="VALID",use_bias=False,use_relu=False)
    return tf.add(conv_1,conv_2)

def blockend(data,num_filters=64):
    conv_1=conv_layer(data,num_filters*4,1,2,padding="VALID",use_bias=False,use_relu=False)
    conv_2=stepConvConvConv(data,num_filters,filter_size=1,strides=1)
    return tf.nn.relu(tf.add(conv_1, conv_2))

def block2(data,num_filters=64):
    print("BLOCK2 #########################")
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)

def block3(data,num_filters=128):
    print("BLOCK3 #########################")
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)
def block4(data,num_filters=256):
    print("BLOCK4 #########################")
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    return blockend(data,num_filters*2)
def block5(data,num_filters=512):
    print("BLOCK5 #########################")
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    conv=stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv, data))
    return pool_avg(data,7,1,'VALID')

def fc_layer(data,num_classes=21):
    print("FC LAYER #########################")
    n,height,width,channels=tf.shape(data).eval()
    data=tf.reshape(data,[-1,height*width*channels])
    # generate weigh [kernal size,kernal size,channel,number of filters]
    w=weight_generater([height*width*channels,num_classes])
    # for each filter W has his  specific bias
    b=bias_generater(num_classes)
    return tf.add(tf.matmul(data,w),b)
    #return tf.nn.softmax(tf.nn.relu(matmul_fc))

def my_model(data,num_classes=21,num_filters=64,filter_size=7,padding='SAME',strides=2):
    conv_1=conv_layer(data,num_filters,filter_size,strides)
    pool_1=pool_max_2x2(conv_1)
    b_1=block1(pool_1)
    b_2=block2(b_1)
    b_3=block3(b_2)
    b_4=block4(b_3)
    b_5=block5(b_4)
    return tf.nn.relu(fc_layer(b_5,num_classes))

sess=tf.InteractiveSession()

"""
# load image
img=cv2.imread("img.tif")
# convetir image to tensor
data=img_to_tensor(img)
# resizing the image
data=tf.image.resize_images(data,[256,256])
# get img dimension
img_shape=tf.shape(data).eval()
print("--input image--\n",img_shape)
# classes
classes=["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential",
           "forest","freeway","golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway","sparseresidential","storagetanks","tenniscourt"]
label=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
"""

# Generate datSets
path="C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
data,labels,classes=datSetGenerator(path)

# get image height,width,channels
height,width,channels=data[1].shape
print("--input image--\n",height,width,channels)

# number of classes
num_classes=len(classes)

x=tf.placeholder(tf.float32,shape=[height,width,channels])
y=tf.placeholder(tf.float32,shape=[num_classes])

logits=my_model(x,num_classes)
softmax=tf.nn.softmax(logits)

cost=tf.reduce_mean(tf.square(softmax - labels[0]))
train=tf.train.GradientDescentOptimizer(0.005).minimize(cost)


correct_prediction=tf.equal(tf.argmax(softmax),tf.argmax(labels[0]))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file=tf.summary.FileWriter("graph/",sess.graph)
    print("logits of model in : \n\n",sess.run(logits,feed_dict={x:data[0]}))
    print("softmax of model in : \n\n",sess.run(softmax,feed_dict={x:data[0]}))
    print(" train : ",sess.run(train,feed_dict={x:data[0]}))
    print("cost : ",sess.run(cost,feed_dict={x:data[0]}))
    print("acc : ",sess.run(acc,feed_dict={x:data[0]}))

print("--- %s seconds ---" % (time.time() - start_time))

sess.close()