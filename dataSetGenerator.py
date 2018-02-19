import numpy as np
from os import listdir
from glob import glob
from time import time
import cv2
import matplotlib.pyplot as plt

def dataSetGenerator(path,resize=False,resize_to=224,percentage=100):
    """

    DataSetsFolder
      |
      |----------class-1
      |        .   |-------image-1
      |        .   |         .
      |        .   |         .
      |        .   |         .
      |        .   |-------image-n
      |        .
      |-------class-n

    :param path: <path>/DataSetsFolder
    :param resize:
    :param resize_to:
    :param percentage:
    :return: images, labels, classes
    """
    start_time = time()
    classes = listdir(path)
    image_list = []
    labels = []
    for classe in classes:
        for filename in glob(path+'/'+classe+'/*.tif'):
            if resize:image_list.append(cv2.resize(cv2.imread(filename),(resize_to, resize_to)))
            else:image_list.append(cv2.imread(filename))
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    indice = np.random.permutation(len(image_list))[:int(len(image_list)*percentage/100)]
    print("\n --- dataSet generated in  %s seconds --- \n" % (np.round(time()-start_time)))
    return np.array([image_list[x] for x in indice]),np.array([labels[x] for x in indice]),np.array(classes)

if __name__ == '__main__':
    # for testing the generator
    path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    data,labels,classes = dataSetGenerator(path,percentage=80)
    print("\n dataSet classes :\n",*classes)
    print("\n   data shape  : ",data.shape)
    print("\n label shape :",labels[100])
    plt.imshow(data[100])
    plt.show()
