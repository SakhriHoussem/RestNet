import tensorflow as tf
import numpy as np
import os
from PIL import Image
import glob

def datSetGenerator(path):
    classes = os.listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob.glob(path+'/'+classe+'/*.tif'): #assuming gif
            image_list.append(np.asarray(Image.open(filename),np.float32))
            label=np.zeros(len(classes),dtype=int)
            label[classes.index(classe)]=1.
            labels.append(label)
    return  image_list,labels,classes

"""
path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
data,labels,classes = datSetGenerator(path)
print(classes)
print(data[699])
print(labels[699])
"""