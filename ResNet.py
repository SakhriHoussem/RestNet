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

def pool_avg(x,ksize=2,strides=2,padding='SAME'):
    return tf.nn.avg_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding)


def batchNorm_layer(): print("---BatchNorm---")

def scale_layer(): print("-----Scale-----")

#def relu_layer(x):return tf.nn.relu(x)


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
    x_image = tf.reshape(data,[-1,height,width,channels])

    conv = conv2d(x_image,w,strides,padding)

    if use_bias: conv +=b

    batchNorm_layer()

    scale_layer()

    if use_relu: conv = tf.nn.relu(conv)

    return conv

def stepConvConv(data,num_filters=64,filter_size=1,strides=1):
    conv21=conv_layer(data,num_filters,filter_size,strides,padding="VALID",use_bias=False)
    conv22=conv_layer(conv21,num_filters,3,strides,padding='SAME',use_bias=False)
    conv23=conv_layer(conv22,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)
    return conv23

def stepConvConvConv(data,num_filters=128,filter_size=1,strides=1):
    conv21=conv_layer(data,num_filters,filter_size,2,padding="VALID",use_bias=False)
    conv22=conv_layer(conv21,num_filters,3,strides,padding='SAME',use_bias=False)
    conv23=conv_layer(conv22,num_filters*4,filter_size,strides,padding="VALID",use_bias=False,use_relu=False)
    return conv23

def block1(data,num_filters=64):
    print("BLOCK1 #########################")
    conv23 = stepConvConv(data,num_filters)
    convi = conv_layer(data,num_filters*4,1,1,padding="VALID",use_bias=False,use_relu=False)
    return tf.add(conv23, convi)

def blockend(data,num_filters=64):
    conv21=conv_layer(data,num_filters*4,1,2,padding="VALID",use_bias=False,use_relu=False)
    convv = stepConvConvConv(data,num_filters,filter_size=1,strides=1)
    return tf.nn.relu(tf.add(conv21, convv))


def block2(data,num_filters=64):
    print("BLOCK2 #########################")

    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    data = blockend(data,num_filters*2)
    return data



def block3(data,num_filters=128):
    print("BLOCK3 #########################")

    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    data = blockend(data,num_filters*2)
    return data
def block4(data,num_filters=256):
    print("BLOCK4 #########################")

    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    data = blockend(data,num_filters*2)
    return data
def block5(data,num_filters=512):
    print("BLOCK5 #########################")

    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))
    conv23 = stepConvConv(data,num_filters)
    data= tf.nn.relu(tf.add(conv23, data))

    return pool_avg(data,7,1,'VALID')

def fc_layer(data,num_filters=21):
    print("FC LAYER #########################")

    n,height,width,channels=tf.shape(data).eval()

    data = tf.reshape(data,[-1,height*width*channels])
    # generate weigh [kernal size,kernal size,channel,number of filters]
    w=weight_generater([height*width*channels,num_filters])

    # for each filter W has his  specific bias
    b=bias_generater(num_filters)

    matmul_fc = tf.matmul(data,w)+b

    return tf.nn.softmax(tf.nn.relu(matmul_fc))
def my_model(data,num_filters=64,filter_size=7,padding='SAME',strides=2):
    conv1=conv_layer(data,num_filters,filter_size,strides)
    pool1=pool_max_2x2(conv1)
    b1 = block1(pool1)
    b2 = block2(b1)
    b3 = block3(b2)
    b4 = block4(b3)
    b5 = block5(b4)
    return fc_layer(b5)

sess=tf.InteractiveSession()
# load image
img=Image.open("img.tif")
# convetir image to tensor
data=img_to_tensor(img)
# resizing the image
# data=tf.image.resize_images(data,[256,256])
# get img dimension
img_dimension=tf.shape(data).eval()
print("--input image--\n",img_dimension)
# get image height,width,channels
height,width,channels=img_dimension
# classes
classes=["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential",
           "forest","freeway","golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway","sparseresidential","storagetanks","tenniscourt"]
# number of classes
num_classes=len(classes)
x=tf.placeholder(tf.float32,shape=[height*width*channels])
y=tf.placeholder(tf.float32,shape=[None,num_classes])


resnet50= my_model(data)
sess.run(tf.global_variables_initializer())

result=sess.run(resnet50)
print(tf.shape(result).eval())
print(result)
print("index",np.argmax(result))
print(classes[np.argmax(result)])
