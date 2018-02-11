import numpy as np
import os
import glob
import time
import cv2

def datSetGenerator(path,img_size=224):
    start_time = time.time()
    classes = os.listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob.glob(path+'/'+classe+'/*.tif'):
            #image_list.append(cv2.imread(filename))
            image_list.append(cv2.resize(cv2.imread(filename),(img_size, img_size)))
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    print("\n --- dataSet generated in  %s seconds --- \n" % (np.round(time.time() - start_time)))
    return np.array(image_list),np.array(labels),np.array(classes)


if __name__ == '__main__':

    # for testing the generator
    path = "C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/"
    data,labels,classes = datSetGenerator(path)
    print("\n dataSet classes :",classes)
    print("\n data shape :",len(data.shape))
    print("\n labels shape :",labels[9])