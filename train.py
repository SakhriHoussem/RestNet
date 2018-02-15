import tensorflow as tf
from dataSetGenerator import dataSetGenerator
import numpy as np
import matplotlib.pyplot as plt
from os import path as PATH
from ResNet import ResNet50


# Generate datSets
# path="C:/Users/shous/Desktop/UCMerced_LandUse/Images/"
data, labels, classes = dataSetGenerator("C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/")
batch_size = 10
epochs = 10
save_dir = "Save/"
save_file ="dataSaved"

#tf.reset_default_graph()


# get image height, width, channels
batche_num, height, width, channels = data.shape
print("Input image size :", height, width, channels)
with tf.Session() as sess:
    if PATH.isdir(save_dir) and PATH.isfile(save_dir+save_file+".meta") and PATH.isfile(save_dir+"checkpoint"):
        print("file exist")
        saver = tf.train.import_meta_graph(save_dir+save_file+".meta")
        saver.restore(sess, tf.train.latest_checkpoint('Save/'))
        print("data restored")
        graph = tf.get_default_graph()
        #[print(n.name) for n in graph.as_graph_def().node]
        x = graph.get_tensor_by_name("t_picture:0")#vrai
        y = graph.get_tensor_by_name("t_labels:0")#vrai
        #train = graph.get_operation_by_name("Train_op")
        #logists = graph.get_operation_by_name("logits")
        #loss = graph.get_operation_by_name("Loss")
        train = tf.get_collection('train_op')#vrai
        loss = tf.get_collection('loss_op')#vrai
        logists = tf.get_collection('logits_op')#vrai
        errors = tf.get_collection('errors')#vrai
        #sess.run(tf.global_variables_initializer())
    else:
        print("file non exist")
        # number of classes
        num_classes = len(classes)

        x = tf.placeholder(tf.float32, [None, height, width, channels],name='t_picture')
        y = tf.placeholder(tf.float32, [None, num_classes],name='t_labels')

        # sess = tf.InteractiveSession()
        logits=ResNet50(x, num_classes)
        # logits = tf.nn.softmax(logits)
        tf.add_to_collection('logits_op',logits)

        # Define a loss function
        loss=tf.reduce_mean(tf.abs(y-logits),name='Loss')
        tf.add_to_collection('loss_op',loss)

        # loss = tf.nn.softmax_cross_entropy_with_logits_v2 (labels=y, logits=logits)
        # loss=tf.nn.l2_loss(logits - y)
        # loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logits), reduction_indices=1))

        # train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        train = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss,name='Train_op')
        tf.add_to_collection('train_op',train)

        # train = tf.train.AdadeltaOptimizer().minimize(loss)
        # correct_prediction=tf.equal(tf.argmax(softmax), tf.argmax(y))
        # acc=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        errors=[]
        tf.add_to_collection('errors',errors)

    print("training start")
    for _ in range(epochs):
            print("*******************  ", _, "  *******************")
            indice = np.random.permutation(batche_num)
            for i in range(batch_size-1):
                min_batch = indice[i*batch_size:(i+1)*batch_size]
                curr_loss, curr_train = sess.run([loss, train], {x: data[min_batch], y: labels[min_batch]})
                print("Iteration %d loss:\n%s" % (i, curr_loss))
                errors.append(curr_loss)
    print("trainig finished")
    #tf.add_to_collection('errors',errors)
    saver = tf.train.Saver()
    saver.save(sess, save_dir+save_file)
    print("file saved in :",save_dir)
    plt.plot(errors, label="loss")
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    plt.show()
    sess.close()