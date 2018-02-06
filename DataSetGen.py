import os
import glob
import time
import tensorflow as tf
import numpy as np

def labelsGen(path):
    labels = []
    classes = os.listdir(path)
    for classe in classes:
        for _ in glob.glob(path+'/'+classe+'/*.tif'):
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
    return  labels,classes


def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_gif(image_string)
  image = tf.cast(image_decoded,tf.float32)
  image_resized = tf.image.resize_images(image, [256,256])
  return image_resized

def DataSetGen(path):
    start_time = time.time()
    classes = os.listdir(path)
    filenames =[]
    labels =[]
    for classe in classes:
        for filename in glob.glob(path+'/'+classe+'/*.tif'):
            label=np.zeros(len(classes))
            label[classes.index(classe)]=1
            labels.append(label)
            filenames.append(filename)
    [filenames.append(filename) for filename in glob.glob(path+'/*/*.tif')]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    print("\n --- dataSet generated in  %s seconds --- \n" % (time.time() - start_time))
    return dataset.map(_parse_function),labels,classes

def imagesGen(path):
    filenames =[]
    [filenames.append(filename) for filename in glob.glob(path+'/*/*.tif')]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    return dataset.map(_parse_function)


if __name__ == '__main__':

    path = "C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    dataset,labels,classes = DataSetGen(path)

    print("classes------------------------------------------------")
    print(classes)
    print("labels------------------------------------------------")
    print(labels)
    print("dataset ------------------------------------------------")
    print(dataset)

