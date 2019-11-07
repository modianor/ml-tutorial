# *_*coding:utf-8 *_*
#

import numpy as np
import tensorflow as tf

'''
KNN的最核心就是距离度量方式，官方例程给出的是L1范数的例子，
我这里改成了L2范数，也就是我们常说的欧几里得距离度量，
另外，虽然是叫KNN，意思是选取k个最接近的元素来投票产生分类，
但是这里只是用了最近的那个数据的标签作为预测值了。
'''
def loadMNIST():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


def KNN(mnist):
    train_x, train_y = mnist.train.next_batch(5000)
    test_x, test_y = mnist.train.next_batch(200)

    xtr = tf.placeholder(tf.float32, [None, 784])
    xte = tf.placeholder(tf.float32, [784])
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2), reduction_indices=1))

    pred = tf.argmin(distance, 0)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    right = 0
    for i in range(200):
        ansIndex = sess.run(pred, {xtr: train_x, xte: test_x[i, :]})
        print('prediction is ', np.argmax(train_y[ansIndex]))
        print('true value is ', np.argmax(test_y[i]))
        if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):
            right += 1.0
    accracy = right / 200.0
    print(accracy)


if __name__ == "__main__":
    mnist = loadMNIST()
    KNN(mnist)
