'''Use a multi-layer convolution network to classify hand-written digits from the MNIST
dataset.

Requires:
    + numpy
    + scipy (only for the logistic and softmax functions, both are achievable
      using numpy easily.
    + pandas (only for reading in MNIST dataset in csv format)
    + matplotlib (only for graph plotting)
    + networkx (only for visualizing the structure of the network)

The MNIST data in csv format are obtained from: http://pjreddie.com/projects/mnist-in-csv/.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-24 15:16:39.
'''

from __future__ import print_function
import numpy as np
from functools import partial
from scipy.special import softmax
from cnn_vectorized import CNNClassifier, ConvLayer, PoolLayer, FCLayer, FlattenLayer
from utils import readMNIST

TRAIN_DATA_FILE = '/home/guangzhi/datasets/mnist/mnist_train.csv'
TEST_DATA_FILE = '/home/guangzhi/datasets/mnist/mnist_test.csv'

LEARNING_RATE = 0.01            # initial learning rate
LAMBDA = 0.01                  # regularization parameter
EPOCHS = 4                    # training epochs


softmax_a = partial(softmax, axis=1)

# -------------Main---------------------------------
if __name__ == '__main__':

    x_data, y_data = readMNIST(TRAIN_DATA_FILE)
    #idx = np.random.permutation(len(x_data))[:5000]
    #x_data = x_data[idx]
    #y_data = y_data[idx]
    print(x_data.shape)
    print(y_data.shape)

    # reshape inputs
    x_data = x_data.reshape([len(x_data), 28, 28, 1])

    # build the network
    cnn = CNNClassifier()

    layer1 = ConvLayer(f=3, pad=0, stride=1, nc_in=1, nc=10,
            learning_rate=LEARNING_RATE, lam=LAMBDA)  # -> mx26x26x10
    layer2 = PoolLayer(f=2, pad=0, stride=2)  # -> mx13x13x10
    layer3 = ConvLayer(f=5, pad=0, stride=1, nc_in=10, nc=16,
            learning_rate=LEARNING_RATE, lam=LAMBDA)  # -> mx9x9x16
    layer4 = FlattenLayer([9,9,16]) # -> mx1296
    layer5 = FCLayer(n_inputs=1296, n_outputs=100,
            learning_rate=LEARNING_RATE, af=None, lam=LAMBDA) # -> mx100
    layer6 = FCLayer(n_inputs=100, n_outputs=10,
            learning_rate=LEARNING_RATE/10, af=softmax_a, lam=LAMBDA)  # -> mx10

    cnn.add(layer1)
    cnn.add(layer2)
    cnn.add(layer3)
    cnn.add(layer4)
    cnn.add(layer5)
    cnn.add(layer6)

    #cnn.loadWeights('cnn_weights_vectorized.npz')

    import time
    t1=time.time()
    costs = cnn.batchTrain(x_data, y_data, EPOCHS, 512)
    #costs = cnn.stochasticTrain(x_data, y_data, EPOCHS)
    t2=time.time()
    print('time=',t2-t1)

    #cnn.saveWeights('cnn_weights_vectorized.npz')


    n_train = x_data.shape[0]
    yhat_train = cnn.predict(x_data)
    yhat_train = np.argmax(yhat_train, axis=1)
    y_true_train = np.argmax(y_data, axis=1)
    n_correct_train = np.sum(yhat_train == y_true_train)

    print('Accuracy in training set:', n_correct_train/float(n_train)*100.)


    # ------------------Open test data------------------
    x_data_test, y_data_test = readMNIST(TEST_DATA_FILE)

    x_data_test = x_data_test.reshape([len(x_data_test), 28, 28, 1])

    n_test = x_data_test.shape[0]
    yhat_test = cnn.predict(x_data_test)
    yhat_test = np.argmax(yhat_test, axis=1)
    y_true_test = np.argmax(y_data_test, axis=1)
    n_correct_test = np.sum(yhat_test == y_true_test)

    print('Accuracy in test set:', n_correct_test/float(n_test)*100.)

    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(12, 10), dpi=100)
    ax = figure.add_subplot(111)
    ax.plot(costs)
    figure.show()
