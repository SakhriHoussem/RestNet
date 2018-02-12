import numpy as np
from os import listdir
from glob import glob
from time import time
import cv2

def dataSetGenerator(path,resize=False,resize_to=224):
    start_time = time()
    classes = listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob(path+'/'+classe+'/*.tif'):
            if resize: image_list.append(cv2.resize(cv2.imread(filename),(resize_to, resize_to)))
            else: image_list.append(cv2.imread(filename))
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    print("\n --- dataSet generated in  %s seconds --- \n" % (np.round(time() - start_time)))
    return np.array(image_list,float),np.array(labels,float),np.array(classes)

if __name__ == '__main__':
    # for testing the generator
    path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    data,labels,classes = dataSetGenerator(path,True)
    print("\n dataSet classes :\n",*classes)
    print("\n data shape :",data.shape)
    print("\n label shape :",labels.shape)
