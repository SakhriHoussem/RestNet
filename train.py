import tensorflow as tf
from dataSetGenerator import dataSetGenerator
import numpy as np
import matplotlib.pyplot as plt

data, labels, classes = dataSetGenerator("C:/Users/shous/Desktop/UCMerced_LandUse - train/Images/")
tf.reset_default_graph()
saver = tf.train.import_meta_graph("Save/dataSaved.meta")
batch_size = 10
epochs = 1
errors = []
batche_num=data.shape[0]
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('Save/'))
    # saver.restore(sess, tf.train.latest_checkpoint('Save/'))

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
    saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer())
    for _ in range(epochs):
            print("*******************  ", _, "  *******************")
            indice = np.random.permutation(batche_num)
            for i in range(batch_size-1):
                min_batch = indice[i*batch_size:(i+1)*batch_size]
                curr_loss, curr_train = sess.run([loss, train], {x: data[min_batch], y: labels[min_batch]})
                print(_, "-Iteration %d\nloss:\n%s" % (i, curr_loss))
                errors.append(curr_loss)
    saver.save(sess, "Save/dataSaved")
    plt.plot(errors, label="loss")
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    plt.show()
    sess.close()