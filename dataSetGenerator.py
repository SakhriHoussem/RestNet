import numpy as np
import os
import cv2
import glob
import time

def datSetGenerator(path):
    start_time = time.time()
    classes = os.listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob.glob(path+'/'+classe+'/*.tif'):
            image_list.append(cv2.imread(filename))
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    print("\n --- dataSet generated in  %s seconds --- \n" % (time.time() - start_time))
    return  np.array(image_list),np.array(labels),np.array(classes)
"""
# for testing the generator
path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
test,tlabels,classes = datSetGenerator(path)
print("\n dataSet classes :\n",classes.shape)
print("\n data shape :",test.shape)
print("\n labels shape :",tlabels.shape)
"""