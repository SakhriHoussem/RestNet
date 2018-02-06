import numpy as np
import os
import glob
import time
import cv2
import tensorflow as tf

def img_to_tensor(img):
    return tf.convert_to_tensor(img,np.float32)
def datSetGenerator(path):
    start_time = time.time()
    classes = os.listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob.glob(path+'/'+classe+'/*.tif'):
            image_list.append(cv2.resize(cv2.imread(filename),(224, 224)))
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    print("\n --- dataSet generated in  %s seconds --- \n" % (time.time() - start_time))
    return np.array(image_list),np.array(labels),np.array(classes)


if __name__ == '__main__':

    # for testing the generator
    path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    data,labels,classes = datSetGenerator(path)
    print("\n dataSet classes :",classes.shape)
    print("\n data shape :",data.shape)
    print("\n labels shape :",labels.shape)

