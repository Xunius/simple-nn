'''Use a multi-layer network to classify hand-written digits from the MNIST
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
Update time: 2021-05-13 13:11:46.
'''

import numpy as np
import matplotlib.pyplot as plt
from multiclassnn import readData, neuralNetwork, softmax_a, plotResult


# ---------------------Globals---------------------
TRAIN_DATA_FILE = '/home/guangzhi/datasets/mnist/mnist_train.csv'
TEST_DATA_FILE = '/home/guangzhi/datasets/mnist/mnist_test.csv'

N_INPUTS = 784                 # input number of units
N_HIDDEN = [200, 100, 50]      # hidden layer sizes
N_OUTPUTS = 10                 # output number of units
LEARNING_RATE = 0.1            # initial learning rate
LAMBDA = 0.01                  # regularization parameter
EPOCHS = 50                    # training epochs


if __name__ == '__main__':

    # -----------------Open MNIST data-----------------
    x_data, y_data = readData(TRAIN_DATA_FILE)
    print('x_data.shape:', x_data.shape)
    print('y_data.shape:', y_data.shape)

    # ------------------Create network------------------
    nn = neuralNetwork(N_INPUTS, N_HIDDEN, N_OUTPUTS, LEARNING_RATE,
                       af_last=softmax_a, lam=LAMBDA)

    # -----------------Mini-batch train-----------------
    nn.plotNN()
    nn.loadWeights('./weights.npz')

    import time
    t1=time.time()
    costs = nn.miniBatchTrain(x_data, y_data, EPOCHS, 64)
    t2=time.time()
    print('time=',t2-t1)
    #costs = nn.batchTrain(x_data, y_data, EPOCHS)
    #costs = nn.stochasticTrain(x_data, y_data, EPOCHS)

    # -----------Plot a graph of the network-----------
    fig, ax = plt.subplots()
    ax.plot(costs, 'b-o')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cost')
    fig.show()

    nn.saveWeights('./weights.npz')

    # ---------------Validate on train set---------------
    yhat_train = np.argmax(nn.predict(x_data), axis=1)
    y_true_train = np.argmax(y_data, axis=1)
    n_correct_train = np.sum(yhat_train == y_true_train)

    print('Training samples:')
    n_test_train = 20
    for ii in np.random.randint(0, len(y_data)-1, n_test_train):
        yhatii = yhat_train[ii]
        ytrueii = y_true_train[ii]
        print('yhat = %d, yii = %d' % (yhatii, ytrueii))

    # ------------------Open test data------------------
    x_data_test, y_data_test = readData(TEST_DATA_FILE)

    # ---------------test on test set---------------
    yhat_test = np.argmax(nn.predict(x_data_test), axis=1)
    y_true_test = np.argmax(y_data_test, axis=1)
    n_correct_test = np.sum(yhat_test == y_true_test)

    print('Test samples:')
    n_test = 20
    for ii in np.random.randint(0, len(y_data_test)-1, n_test):
        yhatii = yhat_test[ii]
        ytrueii = y_true_test[ii]
        print('yhat = %d, yii = %d' % (yhatii, ytrueii))

    print('Accuracy in training set:', n_correct_train/float(len(x_data))*100.)
    print('Accuracy in test set:', n_correct_test/float(len(x_data_test))*100.)

    #----------------Plot some results----------------
    nrows = 3
    ncols = 4
    n_plots =  nrows*ncols
    fig = plt.figure(figsize=(12, 10))
    for i in range(n_plots):
        axi = fig.add_subplot(nrows, ncols, i+1)
        axi.axis('off')
        xi = x_data_test[i]
        yi = y_data_test[i]
        yhati = yhat_test[i]
        plotResult(xi, yi, yhati, ax=axi)

    fig.show()

