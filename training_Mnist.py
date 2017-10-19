import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def mnist_softmax():
    # plot datas
    TIMES = 20000
    x1 = np.arange(0, TIMES)
    y1 = []
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    sess = tf.InteractiveSession()

    # モデルの作成
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 損失とオプティマイザーを定義
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 訓練
    tf.initialize_all_variables().run()
    for i in range(TIMES):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        y1.append(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
        #if i % 10 == 0:
            #print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

    plt.plot(x1, y1)
    plt.show()


def trainingMnist():
    # number of inputs
    input_len = 100 
    # number of classes 
    classes_num = 2

    sess = tf.Session()

    x = tf.placeholder("float", [None, input_len])
    y_ = tf.placeholder("float", [None, classes_num])

    weights1 = tf.Variable(tf.truncated_normal([input_len, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))

    weights3 = tf.Variable(tf.truncated_normal([25, classes_num], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([classes_num]))

    # This time we introduce a single hidden layer into our model...
    hidden_layer_1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

    cost = -tf.reduce_sum(y_*tf.log(model))

    training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.initialize_all_variables()
    sess.run(init)

    for ii in range(10000):
        # y_ -> element of correct class only must be 1
        if ii % 2 == 0: # even number [1,2] => 0
            sess.run(training_step, feed_dict={x: [[1, 2]], y_: [[1, 0]]})
        else:           # odd number  [2,1] => 1
            sess.run(training_step, feed_dict={x: [[2, 1]], y_: [[0, 1]]})

    # prediction
    print("result of prediction --------")
    pred_rslt = sess.run(tf.argmax(model, 1), feed_dict={x: [[1, 2]]})
    print("  input: [1,2] =>" + str(pred_rslt))
    pred_rslt = sess.run(tf.argmax(model, 1), feed_dict={x: [[2, 1]]})
    print("  input: [2,1] =>" + str(pred_rslt))

if __name__ == '__main__':
    # trainingMnist()
    mnist_softmax()
