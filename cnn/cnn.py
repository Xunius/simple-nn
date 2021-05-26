'''Convolutional neural network using numpy+scipy, serial version.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-14 10:50:40.
'''

from __future__ import print_function
import pickle
import numpy as np
from pooling import poolingOverlap, unpooling
from conv3d import padArray, conv3D3, fullConv3D
from utils import getBatch, ReLU, dReLU, crossEntropy


def force2D(x):
    '''Force rank 1 array to 2d to get a column vector'''
    return np.reshape(x, (x.shape[0], -1))

def force3D(x):
    return np.atleast_3d(x)


class ConvLayer(object):
    def __init__(self, f, pad, stride, nc_in, nc, learning_rate, af=None, lam=0.01,
            clipvalue=0.5):
        '''Convolutional layer

        Args:
            f (int): kernel size for height and width.
            pad (int): padding on each edge.
            stride (int): convolution stride.
            nc_in (int): number of channels from input layer.
            nc (int): number of channels this layer.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.

        The layer has <nc> channels/filters. Each filter has shape (f, f, nc_in).
        The <nc> filters are saved in a list `self.filters`, therefore `self.filters[0]`
        corresponds to the 1st filter.

        Bias is saved in `self.biases`, which is a 1d array of length <nc>.
        '''

        self.type = 'conv'
        self.f = f
        self.pad = pad
        self.stride = stride
        self.lr = learning_rate
        self.nc_in = nc_in
        self.nc = nc
        if af is None:
            self.af = ReLU
        else:
            self.af = af
        self.lam = lam
        self.clipvalue = clipvalue

        self.init()

    def init(self):
        '''Initialize weights

        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        np.random.seed(100)
        std = np.sqrt(2/self.f**2/self.nc_in)
        self.filters = np.array([
            np.random.normal(0, scale=std, size=[self.f, self.f, self.nc_in])
            for i in range(self.nc)])
        self.biases = np.random.normal(0, std, size=self.nc)

    @property
    def n_params(self):
        '''Number of parameters in layer'''
        n_filters = self.filters.size
        n_biases = self.nc
        return n_filters + n_biases

    def forward(self, x):
        '''Forward pass of a single image input'''

        x = force3D(x)
        x = padArray(x, self.pad, self.pad)
        # input size:
        ny, nx, nc = x.shape
        # output size:
        oy = (ny+2*self.pad-self.f)//self.stride + 1
        ox = (nx+2*self.pad-self.f)//self.stride + 1
        oc = self.nc
        weight_sums = np.zeros([oy, ox, oc])

        # loop through filters
        for ii in range(oc):
            slabii = conv3D3(x, self.filters[ii], stride=self.stride, pad=0)
            weight_sums[:, :, ii] = slabii[:, :]

        # add bias
        weight_sums = weight_sums+self.biases
        # activate func
        activations = self.af(weight_sums)

        return weight_sums, activations

    def backPropError(self, delta_in, z):
        '''Back-propagate errors

        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.

        The theoretical equation for error back-propagation is:

            \delta^{(l)} = \delta^{(l+1)} \bigotimes_f Rot(W^{(l+1)}) \bigodot f'(z^{(l)})

        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            \bigotimes_f : convolution in full mode.
            Rot() : is rotating the filter by 180 degrees, i.e. a kernel flip.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.

        Computation in practice is more complicated than the above equation.
        '''

        # number of channels of input to layer l weights
        nc_pre = z.shape[-1]
        # number of channels of output from layer l weights
        nc_next = delta_in.shape[-1]

        result = np.zeros_like(z)
        # loop through channels in layer l
        for ii in range(nc_next):
            # flip the kernel
            kii = self.filters[ii, ::-1, ::-1, ...]
            deltaii = delta_in[:, :, ii]
            # loop through channels of input
            for jj in range(nc_pre):
                slabij = fullConv3D(deltaii, kii[:, :, jj], self.stride)
                result[:, :, jj] += slabij

        result = result*dReLU(z)

        return result

    def computeGradients(self, delta, act):
        '''Compute gradients of cost wrt filter weights

        Args:
            delta (ndarray): errors in filter ouputs.
            act (ndarray): activations fed into filter.
        Returns:
            grads (ndarray): gradients of filter weights.
            grads_bias (ndarray): 1d array, gradients of biases.

        The theoretical equation of gradients of filter weights is:

            \partial J / \partial W^{(l)} = a^{(l-1)} \bigotimes \delta^{(l)}

        where:
            J : cost function of network.
            W^{(l)} : weights in filter.
            a^{(l-1)} : activations fed into filter.
            \bigotimes : convolution in valid mode.
            \delta^{(l)} : errors in the outputs from the filter.

        Computation in practice is more complicated than the above equation.
        '''

        nc_out = delta.shape[-1]   # number of channels in outputs
        nc_in = act.shape[-1]      # number of channels in inputs

        grads = np.zeros_like(self.filters)

        for ii in range(nc_out):
            deltaii = np.take(delta, ii, axis=-1)
            gii = grads[ii]
            for jj in range(nc_in):
                actjj = act[:, :, jj]
                gij = conv3D3(actjj, deltaii, stride=1, pad=0)
                gii[:, :, jj] += gij
            grads[ii] = gii

        # gradient clip
        gii = np.clip(gii, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=(0, 1))  # 1d

        return grads, grads_bias

    def gradientDescent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''

        self.filters = self.filters * (1 - self.lr * self.lam/m) - self.lr * grads/m
        self.biases = self.biases-self.lr*grads_bias/m

        return


class PoolLayer(object):
    def __init__(self, f, pad, stride, method='max'):
        '''Pooling layer

        Args:
            f (int): kernel size for height and width.
            pad (int): padding on each edge.
            stride (int): pooling stride. Required to be the same as <f>, i.e.
                non-overlapping pooling.
        Keyword Args:
            method (str): pooling method. 'max' for max-pooling.
                'mean' for average-pooling.
        '''

        self.type = 'pool'
        self.f = f
        self.pad = pad
        self.stride = stride
        self.method = method

        if method != 'max':
            raise Exception("Method %s not implemented" % method)

        if self.f != self.stride:
            raise Exception("Use equal <f> and <stride>.")

    @property
    def n_params(self):
        return 0

    def forward(self, x):
        '''Forward pass'''

        x = force3D(x)
        result, max_pos = poolingOverlap(x, self.f, stride=self.stride,
            method=self.method, pad=False, return_max_pos=True)
        self.max_pos = max_pos  # record max locations

        return result, result

    def backPropError(self, delta_in, z):
        '''Back-propagate errors

        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.

        For max-pooling, each error in <delta_in> is assigned to where it came
        from in the input layer, and other units get 0 error. This is achieved
        with the help of recorded maximum value locations.

        For average-pooling, the error in <delta_in> is divided by the kernel
        size and assigned to the whole pooling block, i.e. even distribution
        of the errors.
        '''

        result = unpooling(delta_in, self.max_pos, z.shape, self.stride)
        return result

    def gradientDescent(self, grads, grads_bias, m):
        '''No weights to learn in pooling layer'''
        return

class FlattenLayer(object):
    def __init__(self, input_shape):
        '''Flatten layer'''

        self.type = 'flatten'
        self.input_shape = input_shape

    @property
    def n_params(self):
        return 0

    def forward(self, x):
        '''Forward pass'''

        x = x.flatten()
        return x, x

    def backPropError(self, delta_in, z):
        '''Back-propagate errors
        '''

        result = np.reshape(delta_in, tuple(self.input_shape))
        return result

    def gradientDescent(self, grads, grads_bias, m):
        '''No weights to learn in flatten layer'''
        return

class FCLayer(object):

    def __init__(self, n_inputs, n_outputs, learning_rate, af=None, lam=0.01,
            clipvalue=0.5):
        '''Fully-connected layer

        Args:
            n_inputs (int): number of inputs.
            n_outputs (int): number of layer outputs.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.
        '''

        self.type = 'fc'
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.clipvalue = clipvalue

        if af is None:
            self.af = ReLU
        else:
            self.af = af

        self.lam = lam   # regularization parameter
        self.init()

    @property
    def n_params(self):
        return self.n_inputs*self.n_outputs + self.n_outputs

    def init(self):
        '''Initialize weights

        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        std = np.sqrt(2/self.n_inputs)
        np.random.seed(100)
        self.weights = np.random.normal(0, std, size=[self.n_outputs, self.n_inputs])
        self.biases = np.random.normal(0, std, size=self.n_outputs)

    def forward(self, x):
        '''Forward pass'''

        z = np.dot(self.weights, x)+self.biases
        a = self.af(z)

        return z, a

    def backPropError(self, delta_in, z):
        '''Back-propagate errors

        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.

        The theoretical equation for error back-propagation is:

            \delta^{(l)} = W^{(l+1)}^{T} \cdot \delta^{(l+1)} \bigodot f'(z^{(l)})

        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.
        '''

        result = np.dot(self.weights.T, delta_in)*dReLU(z)

        return result

    def computeGradients(self, delta, act):
        '''Compute gradients of cost wrt weights

        Args:
            delta (ndarray): errors in ouputs.
            act (ndarray): activations fed into weights.
        Returns:
            grads (ndarray): gradients of weights.
            grads_bias (ndarray): 1d array, gradients of biases.

        The theoretical equation of gradients of filter weights is:

            \partial J / \partial W^{(l)} = \delta^{(l)} \cdot a^{(l-1)}^{T}

        where:
            J : cost function of network.
            W^{(l)} : weights in layer.
            a^{(l-1)} : activations fed into the weights.
            \delta^{(l)} : errors in the outputs from the weights.

        When implemented, had some trouble getting the shape correct. Therefore
        used einsum().
        '''
        #grads = np.einsum('ij,kj->ik', delta, act)
        grads = np.outer(delta, act)
        # gradient-clip
        grads = np.clip(grads, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=-1)

        return grads, grads_bias

    def gradientDescent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''

        self.weights = self.weights * (1 - self.lr * self.lam/m) - self.lr * grads/m
        self.biases = self.biases-self.lr*grads_bias/m

        return


class CNNClassifier(object):

    def __init__(self, cost_func=None):
        '''CNN classifier

        Keyword Args:
            cost_func (callable or None): cost function. If None, use cross-
                entropy cost.
        '''

        self.layers = []

        if cost_func is None:
            self.cost_func = crossEntropy
        else:
            self.cost_func = cost_func

    @property
    def n_layers(self):
        '''Number of layers in network'''
        return len(self.layers)

    @property
    def n_params(self):
        '''Total number of trainable parameters of all layers in network'''
        result = 0
        for ll in self.layers:
            result += ll.n_params
        return result


    def add(self, layer):
        '''Add new layers to the network

        Args:
            layer (ConvLayer|PoolLayer|FCLayer): a ConvLayer, PoolLayer or FCLayer
                object.
        '''
        self.layers.append(layer)


    def feedForward(self, x):
        '''Forward pass of a single record

        Args:
            x (ndarray): input with shape (h, w) or (h, w, c). h is the image
                height, w the image width and c the number of channels.
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

        activations = {0: x}
        weight_sums = {0: x}
        a1 = x
        for ii in range(self.n_layers):
            lii = self.layers[ii]
            zii, aii = lii.forward(a1)
            activations[ii+1] = aii
            weight_sums[ii+1] = zii
            a1 = aii

        return weight_sums, activations

    def feedBackward(self, weight_sums, activations, y):
        '''Backward propogation for a single record

        Args:
            weight_sums (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                weighted sums of each layer.
            activations (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                    activations in each layer.
            y (ndarray): label in shape (m,). m is the number of
                final output units.
        Returns:
            grads (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the weight matrix in each layer.
            grads_bias (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the bias in each layer.
        '''

        delta = activations[self.n_layers] - y
        grads={}
        grads_bias={}

        for jj in range(self.n_layers, 0, -1):
            layerjj = self.layers[jj-1]
            if layerjj.type in ['fc', 'conv']:
                gradjj, biasjj = layerjj.computeGradients(delta, activations[jj-1])
                grads[jj-1]=gradjj
                grads_bias[jj-1]=biasjj

            delta = layerjj.backPropError(delta, weight_sums[jj-1])

        return grads, grads_bias


    def sampleCost(self, yhat, y):
        '''Cost of a single training sample

        Args:
            yhat (ndarray): prediction in shape (m,). m is the number of
                final output units.
            y (ndarray): label in shape (m,).
        Returns:
            cost (float): summed cost.
        '''
        j = self.cost_func(yhat, y)
        return j

    def regCost(self):
        '''Cost from the regularization term

        Defined as the summed squared weights in all layers, not including
        biases.
        '''
        j = 0
        for lii in self.layers:
            if hasattr(lii, 'filters'):
                wii = lii.filters
                jii = np.sum([np.sum(ii**2) for ii in wii])
            elif hasattr(lii, 'weights'):
                wii = lii.weights
                jii = np.sum(wii**2)
            j = j+jii

        return j


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

        for ii, layerii in enumerate(self.layers):
            if layerii.type in ['fc', 'conv']:
                gradii = grads[ii]
                grad_biasii = grads_bias[ii]
                layerii.gradientDescent(gradii, grad_biasii, n)
        return

    def stochasticTrain(self, x, y, epochs):
        '''Stochastic training

        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''
        costs = []
        m = len(x)
        for ee in range(epochs):
            idxs = np.random.permutation(m)
            for ii in idxs:
                xii = np.atleast_3d(x[ii])
                yii = y[ii]
                weight_sums, activations = self.feedForward(xii)
                gradsii, grads_biasii = self.feedBackward(weight_sums, activations, yii)
                self.gradientDescent(gradsii, grads_biasii, 1)

            je = self.evaluateCost(x, y)
            print('# <stochasticTrain>: cost at epoch %d, j = %f' % (ee, je))
            costs.append(je)

        return np.array(costs)

    def sumGradients(self, g1, g2):
        '''Add gradients in two dicts'''
        result=dict([(k, g1[k]+g2[k]) for k in g1.keys()])
        return result

    def batchTrain(self, x, y, epochs, batch_size):
        '''Training using mini batches

        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
            batch_size (int): mini-batch size.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''
        costs = []
        m = len(x)
        for ee in range(epochs):
            batches = getBatch(m, batch_size, randomize=True)
            for idxjj in batches:
                for idxii in idxjj:
                    xii = np.atleast_3d(x[idxii])
                    yii = y[idxii]
                    weight_sums, activations = self.feedForward(xii)
                    gradsii, grads_biasii = self.feedBackward(weight_sums, activations, yii)

                    if idxii == idxjj[0]:
                        gradsjj = gradsii
                        grads_biasjj = grads_biasii
                    else:
                        gradsjj = self.sumGradients(gradsjj, gradsii)
                        grads_biasjj = self.sumGradients(grads_biasjj, grads_biasii)

                self.gradientDescent(gradsjj, grads_biasjj, batch_size)

            je = self.evaluateCost(x, y)
            print('# <batchTrain>: cost at epoch %d, j = %f' % (ee, je))
            costs.append(je)

        return np.array(costs)


    def predict(self, x):
        '''Model prediction

        Args:
            x (ndarray): input with shape (h, w) or (h, w, c). h is the image
                height, w the image width and c the number of channels.
        Returns:
            yhat (ndarray): prediction with shape (m,). m is the number of output units.
        '''

        x = np.atleast_3d(x)
        weight_sums, activations = self.feedForward(x)
        yhat = activations[self.n_layers]
        return yhat


    def evaluateCost(self, x, y):
        '''Compute mean cost on a dataset

        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
        Returns:
            j (float): mean cost over dataset <x, y>.
        '''
        j = 0
        n = len(x)
        for ii in range(n):
            yhatii = self.predict(x[ii])
            yii = y[ii]
            jii = self.sampleCost(yhatii, yii)
            j += jii
        j2 = self.regCost()
        j += j2
        return j/n

    def saveWeights(self, outfilename):
        '''Save model parameters to file

        Args:
            outfilename (str): absolute path to file to save model parameters.

        Parameters are saved using numpy.savez(), loaded using numpy.load().
        '''

        print('\n# <saveWeights>: Save network weights to file', outfilename)

        dump = {}
        for ii, layerii in enumerate(self.layers):
            if layerii.type == 'conv':
                dii = {'lr': layerii.lr,
                       'lam': layerii.lam,
                       'filters': layerii.filters,
                       'biases': layerii.biases}
            elif layerii.type == 'fc':
                dii = {'lr': layerii.lr,
                       'lam': layerii.lam,
                       'weights': layerii.weights,
                       'biases': layerii.biases}
            dump[str(ii)] = dii

        with open(outfilename, 'wb') as fout:
            pickle.dump(dump, fout)

        return

    def loadWeights(self, abpathin):
        '''Load model parameters from file

        Args:
            abpathin (str): absolute path to file to load model parameters.

        Parameters are saved using numpy.savez(), loaded using numpy.load().
        '''

        print('\n# <saveWeights>: Load network weights from file', abpathin)

        with open(abpathin, 'rb') as fin:
            params = pickle.load(fin)
            layer_keys = list(params.keys())
            for ii, layerii in enumerate(self.layers):
                if str(ii) in layer_keys:
                    dii = params[str(ii)]
                    layerii.lr = dii['lr']
                    layerii.lam = dii['lam']
                    layerii.biases = dii['biases']
                    if layerii.type == 'fc':
                        layerii.weights = dii['weights']
                    elif layerii.type == 'conv':
                        layerii.filters = dii['filters']

        return





