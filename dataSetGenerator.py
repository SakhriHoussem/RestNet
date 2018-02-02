import tensorflow as tf
import numpy as np
import os
from PIL import Image
import glob

def img_to_tensor(img):
    return tf.convert_to_tensor(np.asarray(img,np.float32),np.float32)
def datSetGenerator(path):
    classes = os.listdir(path)
    print(len(classes))
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob.glob(path+'/'+classes[1]+'/*.tif'): #assuming gif
            im=Image.open(filename)
            label=np.zeros(len(classes),dtype=int)
            label[classes.index(classe)]=1
            image_list.append([img_to_tensor(im)])
            labels.append(label)
    return  image_list,labels

path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
image_list,labels = datSetGenerator(path)
print(image_list[756],labels[756])