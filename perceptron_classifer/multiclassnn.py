'''Neural network for multi-class classification using numpy+scipy

Requires:
    + numpy
    + scipy (only for the logistic and softmax functions, both are achievable
      using numpy easily.
    + networkx (only for visualizing the structure of the network)

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-13 13:11:46.
'''

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.special import expit, softmax
import pandas as pd


softmax_a = partial(softmax, axis=1)


def readData(path):
    '''Read csv formatted MNIST data

    Args:
        path (str): absolute path to input csv data file.
    Returns:
        x_data (ndarray): mnist input array in shape (n, 784), n is the number
            of records, 784 is the number of (flattened) pixels (28 x 28) in
            each record.
        y_data (ndarray): mnist label array in shape (n, 10), n is the number
            of records. Each label is a 10-element binary array where the only
            1 in the array denotes the label of the record.
    '''
    data_file = pd.read_csv(path, header=None)
    x_data = []
    y_data = []
    for ii in range(len(data_file)):
        lineii = data_file.iloc[ii, :]
        imgii = np.array(lineii.iloc[1:])/255.
        labelii = np.zeros(10)
        labelii[int(lineii.iloc[0])] = 1
        y_data.append(labelii)
        x_data.append(imgii)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def plotResult(x_data, y_data, yhat, ax=None):
    '''Plot an image and prediction

    Args:
        x_data (ndarray): input image array in shape (784,).
        y_data (ndarray): mnist label array in shape (10,).
        yhat (int): predicted digit.
    Keyword Args:
        ax (matplotlib axis): if None, create a new.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    nl = int(np.sqrt(x_data.size))
    img = x_data.reshape(nl, nl)
    ax.imshow(img)
    ax.set_title('Input = %d, predict = %d' % (int(np.argmax(y_data)), yhat))
    if ax is None:
        fig.show()

    return


def crossEntropy(yhat, y):
    '''Cross entropy cost function '''
    eps = 1e-10
    yhat = np.clip(yhat, eps, 1-eps)
    return -np.nansum(y*np.log(yhat))


def dlogistic(x):
    '''Derivative of the logistic function'''
    return np.exp(-x)/(1 + np.exp(-x))**2


class neuralNetwork(object):
    def __init__(
            self, n_inputs, n_hidden, n_outputs, learning_rate, af=None,
            daf=None, af_last=None, init_func='xavier', cost_func=None,
            lam=0.01):
        '''Neural network

        Args:
            n_inputs (int): number of input units. This is the number of units
                in the 1st layer (indexed 0).
            n_hidden (list/tuple/int): if list/tuple of integers, the number of units
                in each hidden layer. If int, number of units in the single hidden
                layer.
            n_outputs (int): number of output units.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callabel or None): activation function. If None, use the logistic
                function as default.
            daf (callabel or None): derivative of the activation function.
                If None, use the derivative of the logistic function as default.
            af_last (callable or None): activation function for the output
                layer. If None, use softmax as default.
            init_func (str): initialization method. One of 'xavier', 'he'.
            cost_func (callable or None): cost function. If None, use cross-
                entropy cost.
            lam (float): regularization parameter.

        The functions of af, daf and af_last all assume a signature of
        a = g(z).
        '''

        if np.isscalar(n_hidden):
            n_hidden = [n_hidden, ]

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_nodes = [n_inputs, ]+n_hidden+[n_outputs, ]
        self.n_layers = len(self.n_nodes)
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.init_func = init_func
        self.lam = lam

        if af is None:
            self.af = expit
            self.daf = dlogistic
        else:
            self.af = af
            self.daf = daf

        if af_last is None:
            self.af_last = softmax_a
        else:
            self.af_last = af_last

        if cost_func is None:
            self.cost_func = crossEntropy
        else:
            self.cost_func = cost_func

        # initialize weights
        self.init()

    def init(self):

        self.thetas = {}  # theta_l is mapping from layer l-1 to l
        self.biases = {}  # bias_l is added to layer l when mapping from layer l-1 to l

        for ii in range(1, self.n_layers):
            if self.init_func == 'xavier':
                stdii = np.sqrt(6/(self.n_nodes[ii-1]+self.n_nodes[ii]))
                thetaii = (np.random.rand(self.n_nodes[ii], self.n_nodes[ii-1]) - 0.5) * stdii
            elif self.init_func == 'he':
                stdii = np.sqrt(2/self.n_nodes[ii-1])
                thetaii = np.random.normal(0, stdii, size=(self.n_nodes[ii],
                    self.n_nodes[ii-1]))

            self.thetas[ii] = thetaii
            self.biases[ii] = np.zeros([1, self.n_nodes[ii]])

    @staticmethod
    def force2D(x):
        '''Force to row vector for 1d array'''
        return np.atleast_2d(x)

    # ---------------------Forward---------------------
    def feedForward(self, x):
        '''Forward pass

        Args:
            x (ndarray): input data with shape (n, f). f is the number of input units,
                n is the number of records.
        Returns:
            weight_sums (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                weighted sums of each layer:

                    z^{(l+1)} = a^{(l)} \cdot \theta^{(l+1)}^{T} + b^{(l+1)}

                where:
                    z^{(l+1)}: weighted sum in layer l+1.
                    a^{(l)}: activation in layer l.
                    \theta^{(l+1)}: weights that map from layer l to l+1.
                    b^{(l+1)}: biases added to layer l+1.

                The value for key=0 is the same as input <x>.
            activations (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                    activations in each layer. See above.
        '''
        x = self.force2D(x)
        activations = {0: x}
        weight_sums = {0: x}
        a1 = x
        for ii in range(1, self.n_layers):
            bii = self.biases[ii]
            zii = np.einsum('ij,kj->ik', a1, self.thetas[ii]) + bii
            if ii == self.n_layers-1:
                aii = self.af_last(zii)
            else:
                aii = self.af(zii)
            activations[ii] = aii
            weight_sums[ii] = zii
            a1 = aii

        return weight_sums, activations

    def sampleCost(self, yhat, y):
        '''Cost of a single training sample/batch

        Args:
            yhat (ndarray): prediction in shape (n, m). m is the number of
                final output units, n is the number of records.
            y (ndarray): label in shape (n, m).
        Returns:
            cost (float): summed cost.
        '''
        return self.cost_func(yhat, y)

    def regulizationCost(self):
        '''Cost from the regularization term

        Defined as the summed squared weights in all layers, not including
        biases.
        '''
        j = 0
        for ll in range(1, self.n_layers):
            j = j+np.sum(self.thetas[ll]**2)
        return j

    def feedBackward(self, weight_sums, activations, y):
        '''Backward propogation

        Args:
            weight_sums (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                weighted sums of each layer.
            activations (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                    activations in each layer.
            y (ndarray): label in shape (m, n). m is the number of
                final output units, n is the number of records.
        Returns:
            grads (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the weight matrix in each layer.
            grads_bias (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the bias in each layer.
        '''
        grads = {}       # gradients for weight matrices
        grads_bias = {}  # gradients for bias
        y = self.force2D(y)
        delta = activations[self.n_layers-1] - y

        for jj in range(self.n_layers-1, 0, -1):
            grads[jj] = np.einsum('ij,ik->jk', delta, activations[jj-1])
            grads_bias[jj] = np.sum(delta, axis=0, keepdims=True)
            if jj != 1:
                delta = np.einsum('ij,jk->ik', delta, self.thetas[jj]) *\
                        self.daf(weight_sums[jj-1])

        return grads, grads_bias

    # -----------------Gradient descent-----------------
    def gradientDescent(self, grads, grads_bias, n):
        '''Perform gradient descent parameter update

        Args:
            grads (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the weight matrix in each layer.
            grads_bias (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the bias in each layer.
            n (int): number of records contributing to the gradient computation.

        Update rule:

            \theta_i = \theta_i - \alpha (g_i + \lambda \theta_i)

        where:
            \theta_i: weight i.
            \alpha: learning rate.
            \g_i: gradient of weight i.
            \lambda: regularization parameter.
        '''

        n = float(n)
        for jj in range(1, self.n_layers):
            theta_jj = self.thetas[jj]
            theta_jj = theta_jj - self.lr * \
                grads[jj]/n - theta_jj*self.lr*self.lam/n
            self.thetas[jj] = theta_jj

            bias_jj = self.biases[jj]
            bias_jj = bias_jj - self.lr * grads_bias[jj]/n
            self.biases[jj] = bias_jj

        return

    def stochasticTrain(self, x, y, epochs):
        '''Stochastic training

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''
        self.costs = []
        m = len(x)
        for ee in range(epochs):
            idxs = np.random.permutation(m)
            for i, ii in enumerate(idxs):
                xii = x[[ii]]
                yii = y[[ii]]
                weight_sums, activations = self.feedForward(xii)
                grads, grads_bias = self.feedBackward(
                    weight_sums, activations, yii)
                self.gradientDescent(grads, grads_bias, 1)

            yhat = self.predict(x)
            j = self.sampleCost(yhat, y)
            j2 = self.regulizationCost()
            j = j/m + j2*self.lam/m
            print('# <stochasticTrain>: Epoch = %d, Cost = %f' % (ee, j))
            # annealing
            if len(self.costs) > 1 and j > self.costs[-1]:
                self.lr *= 0.9
                print('# <stochasticTrain>: Annealing learning rate, current lr =', self.lr)
            self.costs.append(j)

        self.costs = np.array(self.costs)

        return self.costs

    def _batchTrain(self, x, y):
        '''Training using a single sample or a sample batch

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
        Returns:
            j (float): summed cost over samples in <x> <y>.

        Slow version, use the sample one by one
        '''
        m = len(x)
        grads = dict([(ll, np.zeros_like(self.thetas[ll])) for ll
                      in range(1, self.n_layers)])
        grads_bias = dict([(ll, np.zeros_like(self.biases[ll])) for ll
                           in range(1, self.n_layers)])
        j = 0   # accumulates cost
        idx = np.random.permutation(m)
        for ii in idx:
            xii = x[ii]
            yii = y[ii]
            weight_sums, activations = self.feedForward(xii)
            j = j+self.sampleCost(activations[self.n_layers-1], yii)
            gradii, grads_biasii = self.feedBackward(
                weight_sums, activations, yii)
            for ll in range(1, self.n_layers):
                grads[ll] = grads[ll]+gradii[ll]
                grads_bias[ll] = grads_bias[ll]+grads_biasii[ll]

        self.gradientDescent(grads, grads_bias, m)

        return j

    def _batchTrain2(self, x, y):
        '''Training using a single sample or a sample batch

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
        Returns:
            j (float): summed cost over samples in <x> <y>.

        Vectorized version. About 3x faster than _batchTrain()
        '''
        m = len(x)
        weight_sums, activations = self.feedForward(x)
        grads, grads_bias = self.feedBackward(weight_sums, activations, y)
        self.gradientDescent(grads, grads_bias, m)
        j = self.sampleCost(activations[self.n_layers-1], y)

        return j

    def batchTrain(self, x, y, epochs):
        '''Training using all samples

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''

        self.costs = []
        m = len(x)

        for ee in range(epochs):
            # j=self._batchTrain(x,y)
            j = self._batchTrain2(x, y)
            j2 = self.regulizationCost()
            j = j/m + j2*self.lam/m
            print('# <batchTrain>: Epoch = %d, Cost = %f' % (ee, j))
            # annealing
            if len(self.costs) > 1 and j > self.costs[-1]:
                self.lr *= 0.9
                print('# <batchTrain>: Annealing learning rate, current lr =', self.lr)

            self.costs.append(j)

        self.costs = np.array(self.costs)

        return self.costs

    def miniBatchTrain(self, x, y, epochs, batch_size):
        '''Training using mini batches

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
            batch_size (int): mini-batch size.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''

        self.costs = []
        m = len(x)
        for ee in range(epochs):
            j = 0
            idx = np.random.permutation(m)
            id1 = 0
            while True:
                id2 = id1+batch_size
                id2 = min(id2, m)
                idxii = idx[id1:id2]
                xii = x[idxii]
                yii = y[idxii]
                # j=j+self._batchTrain(xii,yii)
                j = j+self._batchTrain2(xii, yii)
                id1 += batch_size
                if id1 > m-1:
                    break

            j2 = self.regulizationCost()
            j = j/m + j2*self.lam/m

            print('# <miniBatchTrain>: Epoch = %d, Cost = %f' % (ee, j))
            # annealing
            if len(self.costs) > 1 and j > self.costs[-1]:
                self.lr *= 0.9
                print('# <miniBatchTrain>: Annealing learning rate, current lr =', self.lr)

            self.costs.append(j)

        self.costs = np.array(self.costs)

        return self.costs

    def predict(self, x):
        '''Model prediction

        Args:
            x (ndarray): input with shape (n, f). f is the number of input units,
                n is the number of records.
        Returns:
            yhat (ndarray): prediction with shape (n, m). m is the number of output units,
                n is the number of records.
        '''
        yhat = self.feedForward(x)[1][self.n_layers-1]
        return yhat

    def plotNN(self):
        '''Plot structure of network'''

        import networkx as nx

        self.graph = nx.DiGraph()
        show_nodes_half = 3

        layered_nodes = []
        for ll in range(self.n_layers-1):
            # layer l
            nodesll = list(zip([ll, ] * self.n_nodes[ll], range(self.n_nodes[ll])))
            # layer l+1
            nodesll2 = list(zip([ll+1, ]*self.n_nodes[ll+1], range(self.n_nodes[ll+1])))

            # omit if too many
            if len(nodesll) > show_nodes_half*3:
                nodesll = nodesll[:show_nodes_half] + nodesll[-show_nodes_half:]
            if len(nodesll2) > show_nodes_half*3:
                nodesll2 = nodesll2[:show_nodes_half] + nodesll2[-show_nodes_half:]

            # build network
            edges = [(a, b) for a in nodesll for b in nodesll2]
            self.graph.add_edges_from(edges)

            layered_nodes.append(nodesll)
            if ll == self.n_layers-2:
                layered_nodes.append(nodesll2)

        # -----------------Adjust positions-----------------
        pos = {}

        for ll, nodesll in enumerate(layered_nodes):
            posll = np.array(nodesll).astype('float')
            yll = posll[:, 1]
            if self.n_nodes[ll] > show_nodes_half*3:
                bottom = yll[:show_nodes_half]
                top = yll[-show_nodes_half:]
                bottom = bottom/np.ptp(bottom)/3.
                top = (top-np.min(top))/np.ptp(top)/3.+2/3.
                yll = np.r_[bottom, top]
            else:
                yll = yll/np.ptp(yll)

            posll[:, 1] = yll

            pos.update(dict(zip(nodesll, posll)))

        #-------------------Draw figure-------------------
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        nx.draw(self.graph, pos=pos, ax=ax, node_size=1200, with_labels=True,
                node_color='c', alpha=0.7)
        plt.show(block=False)

        return

    def saveWeights(self, outfilename):
        '''Save model parameters to file

        Args:
            outfilename (str): absolute path to file to save model parameters.

        Parameters are saved using numpy.savez(), loaded using numpy.load().
        '''

        print('\n# <saveWeights>: Save network weights to file', outfilename)

        dump = {'lr': self.lr,
                'n_nodes': self.n_nodes,
                'lam': self.lam}
        for ll in range(1, self.n_layers):
            dump['theta_%d' % ll] = self.thetas[ll]
            dump['bias_%d' % ll] = self.biases[ll]

        np.savez(outfilename, **dump)

        return

    def loadWeights(self, abpathin):
        '''Load model parameters from file

        Args:
            abpathin (str): absolute path to file to load model parameters.

        Parameters are saved using numpy.savez(), loaded using numpy.load().
        '''

        print('\n# <saveWeights>: Load network weights from file', abpathin)

        with np.load(abpathin) as npzfile:
            for ll in range(1, self.n_layers):
                thetall = npzfile['theta_%d' % ll]
                biasll = npzfile['bias_%d' % ll]
                self.thetas[ll] = thetall
                self.biases[ll] = biasll
            self.lam = npzfile['lam']
            self.lr = npzfile['lr']

        return


