import tensorflow as tf
from dataSetGenerator import dataSetGenerator
import numpy as np
import matplotlib.pyplot as plt
from os import path as PATH
from os import makedirs
from ResNet import ResNet50
from colors import *

def training(path="/",batch_size = 10,epochs = 30,save_dir = "Saved/",save_file ="dataSaved",sess=tf.Session()):

    # get image height, width, channels
    batche_num, height, width, channels = data.shape
    print(OKBLUE+"Input image size :"+ENDC, height, width, channels)

    if not PATH.isdir(save_dir):
        makedirs(save_dir)
        print(OKGREEN+save_dir,"is created"+ENDC)
    with sess:
        if PATH.isdir(save_dir) and PATH.isfile(save_dir+save_file+".meta") and PATH.isfile(save_dir+"checkpoint"):
            print(OKGREEN+"files are exist"+ENDC)
            saver = tf.train.import_meta_graph(save_dir+save_file+".meta")
            saver.restore(sess, tf.train.latest_checkpoint('Save/'))
            print(OKGREEN+"data are restored"+ENDC)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("t_picture:0")#vrai
            y = graph.get_tensor_by_name("t_labels:0")#vrai
            train = tf.get_collection('train_op')#vrai
            loss = tf.get_collection('loss_op')#vrai
            logists = tf.get_collection('logits_op')#vrai
            errors = tf.get_collection('errors')#vrai
        else:
            print(FAIL+"files are not exist"+ENDC)
            # number of classes
            num_classes = len(classes)
            x = tf.placeholder(tf.float32, [None, height, width, channels],name='t_picture')
            y = tf.placeholder(tf.float32, [None, num_classes],name='t_labels')

            # sess = tf.InteractiveSession()
            logits=ResNet50(x, num_classes)
            softmax = tf.nn.softmax(logits)
            tf.add_to_collection('logits_op',logits)

            # Define a loss function
            loss=tf.reduce_mean(tf.abs(y-logits),name='Loss')
            tf.add_to_collection('loss_op',loss)

            # loss = tf.nn.softmax_cross_entropy_with_logits_v2 (labels=y, logits=logits)
            # loss = tf.nn.l2_loss(logits - y)
            # loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logits), reduction_indices=1))

            # train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
            train = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss,name='Train_op')
            tf.add_to_collection('train_op',train)

            # train = tf.train.AdadeltaOptimizer().minimize(loss)
            correct_prediction=tf.equal(tf.argmax(softmax), tf.argmax(y))
            tf.add_to_collection('prediction_op',correct_prediction)

            acc=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.add_to_collection('acc_op',acc)

            sess.run(tf.global_variables_initializer())
            errors=[]
            tf.add_to_collection('errors',errors)

        print(OKGREEN+"training start"+ENDC)
        for _ in range(epochs):
                print(OKBLUE+"*******************  ", _, "  *******************"+ENDC)
                indice = np.random.permutation(batche_num)
                for i in range(batch_size-1):
                    min_batch = indice[i*batch_size:(i+1)*batch_size]
                    curr_loss, curr_train = sess.run([loss, train], {x: data[min_batch], y: labels[min_batch]})
                    print("Iteration %d loss:\n%s" % (i, curr_loss))
                    errors.append(curr_loss)
        print(OKGREEN+"training is finished"+ENDC)
        saver = tf.train.Saver()
        saver.save(sess, save_dir+save_file)
        print(OKGREEN+"files saved in :"+ENDC,save_dir)
        plt.plot(errors, label="loss")
        plt.xlabel('# epochs')
        plt.ylabel('MSE')
        plt.show()
        sess.close()

if __name__ == '__main__':

    # path="C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
    path="C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/"
    data, labels, classes = dataSetGenerator(path)
    batch_size = 10
    epochs = 30
    save_dir = "Save/"
    save_file = "dataSaved"
    training(path,batch_size,epochs,save_dir,save_file)