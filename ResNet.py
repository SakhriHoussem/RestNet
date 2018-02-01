import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

def img_to_tensor(img):
    return tf.convert_to_tensor(np.asarray(img,np.float32),np.float32)

def weight_generater(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def bias_generater(shape):
    return tf.Variable(tf.constant(.05,shape=[shape]))

def conv2d(x,W,strides=1,padding='SAME'):
    return tf.nn.conv2d(x,W,[1,strides,strides,1],padding='SAME')

def pool_max_2x2(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)



def conv_layer(data,num_filters,filter_size,strides=1,padding="SAME",use_bias=True):

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
    x_image = tf.reshape(data,[-1,height,width,channels])

    conv = conv2d(x_image,w,strides,padding)

    if use_bias: conv +=b

    return conv


def batchNorm_layer(): print("---BatchNorm---")
def scale_layer(): print("-----Scale-----")
def relu_layer(x): return tf.nn.relu(x)

bar = "---------------"
concat ="      ||<<<-------------------||\n      \/"
fleshpara =bar+"\n      ||-----------------------||\n      \/                       ||\n"
flesh =bar+"\n      ||\n      \/\n"+bar
para = "               ||"
sess=tf.InteractiveSession()
# load image
img=Image.open("img.tif")
# convetir image to tensor
data=img_to_tensor(img)
# resizing the image
# data=tf.image.resize_images(data,[256,256])
# get img dimension
img_dimension=tf.shape(data).eval()
print("\n\n--input image--\n",img_dimension)
# get image height,width,channels
height,width,channels=img_dimension
# classes
classes=["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential",
           "forest","freeway","golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway","sparseresidential","storagetanks","tenniscourt"]
# number of classes
num_classes=len(classes)
x=tf.placeholder(tf.float32,shape=[height*width*channels])
y=tf.placeholder(tf.float32,shape=[None,num_classes])
#************************************************************************************
################   block1    ###################
print(flesh)
    #############   conv1    ###################
# filter size
filter_size=7
# number of filters
num_filters=64
strides=2
padding='SAME'
conv1=conv_layer(data,num_filters,filter_size,strides)
print("-----Conv 1----\n",*tf.shape(conv1).eval())
batchNorm_layer()
scale_layer()
conv1=relu_layer(conv1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
result=sess.run(conv1)
print("-----Relu------\n",*tf.shape(result).eval())
################   Pooling    ###################
print(flesh)


pool1= pool_max_2x2(conv1,3)
result=sess.run(pool1)
print("---Pooling 1---\n",*tf.shape(result).eval())
#************************************************************************************
    #############   Block2    ###################
print(fleshpara+bar,para)
    #############   conv21    ###################
# filter size
filter_size=1
# number of filters
num_filters=64
conv21=conv_layer(pool1,num_filters,filter_size,strides=1,padding='VALID',use_bias=False)
print("----Conv 21----",para,"\n ",*tf.shape(conv21).eval())
batchNorm_layer()
scale_layer()
conv21 = relu_layer(conv21)
sess.run(tf.global_variables_initializer())
result=sess.run(conv21)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
   #############   conv22    ###################
# filter size
filter_size=3
# number of filters
num_filters=64
conv22=conv_layer(conv21,num_filters,filter_size,strides=1,padding='SAME',use_bias=False)
print("----Conv 22----",para,"\n ",*tf.shape(conv22).eval())
batchNorm_layer()
scale_layer()
conv22 = relu_layer(conv22)
sess.run(tf.global_variables_initializer())
result=sess.run(conv22)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
#************************************************************************************
    #############   conv23    ###################
# filter size
filter_size=1
# number of filters
num_filters=256
conv23=conv_layer(conv22,num_filters,filter_size,strides=1,padding='VALID',use_bias=False)
# filter size
filter_size=1
# number of filters
num_filters=256
padding='VALID'
#strides=1
conv11=conv_layer(pool1,num_filters,filter_size,1,padding=padding,use_bias=False)
print("----Conv 23----          ----Conv 11----")
print(*tf.shape(conv23).eval(),"              ",*tf.shape(conv11).eval())
batchNorm_layer()
scale_layer()
sess.run(tf.global_variables_initializer())
result=sess.run(conv23)
print(concat)
    #############   res1    ###################
res1 = sess.run(conv23+conv11)
print(bar,"\n------Res1-----\n",*tf.shape(res1).eval())
res1 = relu_layer(res1)
print("------Relu-----\n",*tf.shape(res1).eval())
#************************************************************************************
#     #############   Block3    ###################
print(fleshpara+bar,para)
    #############   conv31    ###################
# filter size
filter_size=1
# number of filters
num_filters=64
conv31=conv_layer(res1,num_filters,filter_size,strides=1,padding='VALID',use_bias=False)
print("----Conv 31----",para,"\n ",*tf.shape(conv21).eval())
batchNorm_layer()
scale_layer()
conv31 = relu_layer(conv31)
sess.run(tf.global_variables_initializer())
result=sess.run(conv31)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
   #############   conv22    ###################
# filter size
filter_size=3
# number of filters
num_filters=64
conv32=conv_layer(conv31,num_filters,filter_size,strides=1,padding='SAME',use_bias=False)
print("----Conv 32----",para,"\n ",*tf.shape(conv32).eval())
batchNorm_layer()
scale_layer()
conv32 = relu_layer(conv32)
sess.run(tf.global_variables_initializer())
result=sess.run(conv32)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
#************************************************************************************
    #############   conv33    ###################
# filter size
filter_size=1
# number of filters
num_filters=256
conv33=conv_layer(conv32,num_filters,filter_size,strides=1,padding='VALID',use_bias=False)
print("----Conv 33----")
print(*tf.shape(conv33).eval())
batchNorm_layer()
scale_layer()
sess.run(tf.global_variables_initializer())
result=sess.run(conv33)
print(concat)
    #############   res1    ###################
res1 = sess.run(conv33+conv11)
print(bar,"\n------Res1-----\n",*tf.shape(res1).eval())
res1 = relu_layer(res1)
print("------Relu-----\n",*tf.shape(res1).eval())
#************************************************************************************
    #############   Block4    ###################
print(fleshpara+bar,para)
   #############   conv41    ###################
# filter size
filter_size=1
# number of filters
num_filters=128
conv41=conv_layer(res1,num_filters,filter_size,strides=2,padding='VALID',use_bias=False)
print("----Conv 41----",para,"\n ",*tf.shape(conv41).eval())
batchNorm_layer()
scale_layer()
conv41 = relu_layer(conv41)
sess.run(tf.global_variables_initializer())
result=sess.run(conv41)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
   #############   conv42    ###################
# filter size
filter_size=3
# number of filters
num_filters=128
conv42=conv_layer(conv41,num_filters,filter_size,strides=1,padding='SAME',use_bias=False)
print("----Conv 42----",para,"\n ",*tf.shape(conv42).eval())
batchNorm_layer()
scale_layer()
conv42 = relu_layer(conv42)
sess.run(tf.global_variables_initializer())
result=sess.run(conv42)
print("-----Relu------",para,"\n ",*tf.shape(result).eval())
print(flesh)
#************************************************************************************


   #############   conv43    ###################
# filter size
filter_size=1
# number of filters
num_filters=512
conv43=conv_layer(conv42,num_filters,filter_size,strides=1,padding='VALID',use_bias=False)
print("----Conv 43----",para)
print(*tf.shape(conv43).eval())
batchNorm_layer()
scale_layer()
sess.run(tf.global_variables_initializer())
result=sess.run(conv43)
print(concat)

sess.close()