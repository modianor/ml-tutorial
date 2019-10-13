import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data

from mnist_cnn_model import inference

datas: Datasets = input_data.read_data_sets(train_dir='./MNIST_DATA', one_hot=True)

train_data: DataSet = datas.train
test_data: DataSet = datas.test

print(train_data.num_examples)
print(test_data.num_examples)

classes = 10
min_batch = 300
epoch = 3
h1_layer, h2_layer, h3_layer = 400, 200, 100
width, height, channels = 28, 28, 1
lr = 0.003
batch_num = int(train_data.num_examples / min_batch)

X = tf.placeholder(dtype=tf.float32, shape=[None, width, height, channels], name='input_X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='output_Y')

softmax_linear = inference(X, min_batch, classes)

result = tf.equal(tf.argmax(softmax_linear, 1), tf.argmax(Y, 1))

acc = tf.reduce_mean(tf.cast(result, tf.float32))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax_linear, labels=Y))

train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('********************************begin to train**************************************')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    for i in range(epoch):
        for j in range(batch_num):
            train_data_x, train_data_y = train_data.next_batch(batch_size=min_batch)
            train_data_x = np.reshape(train_data_x, (min_batch, 28, 28))
            train_data_x = np.expand_dims(train_data_x, axis=-1)
            _, curr_loss, curr_acc = session.run(
                fetches=[train_op, loss_op, acc],
                feed_dict={X: train_data_x, Y: train_data_y}
            )

            if (j + 1) % 5 == 0:
                print('the {} bacth loss and train accuracy : {} , {}'.format(j + 1, curr_loss, curr_acc))

        test_data_x, test_data_y = test_data.next_batch(batch_size=min_batch, shuffle=True)
        test_data_x = np.reshape(test_data_x, (min_batch, 28, 28))
        test_data_x = np.expand_dims(test_data_x, axis=-1)

        _, curr_loss, curr_acc = session.run(
            fetches=[train_op, loss_op, acc],
            feed_dict={X: test_data_x, Y: test_data_y}
        )
        print('====================================================================================')
        print('====================================================================================')
        print('====================================================================================')
        print('====================================================================================')
        print('the {} epoch loss and test accuracy : {} , {}'.format(i + 1, curr_loss, curr_acc))
        print('====================================================================================')
        print('====================================================================================')
        print('====================================================================================')

    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('********************************training finish*************************************')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
    print('====================================================================================')
