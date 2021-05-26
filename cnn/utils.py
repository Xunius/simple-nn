'''General purpose utilities

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-26 11:07:35.
'''

from __future__ import print_function
import numpy as np
import pandas as pd

def getBatch(n, bsize, randomize=False):
    '''Generator to iterate through a range in batches

    Args:
        n (int): range from 0 to <n> to iterate through.
        bsize (int): batch size.
    Keyword Args:
        randomize (bool): if True, iterate through randomly permuated range.
    Returns:
        idxii (list): list of integers with a length <= <bsize>.
    '''

    id1 = 0
    if randomize:
        idx = np.random.permutation(n)
    while True:
        id2 = id1+bsize
        id2 = min(id2, n)
        if randomize:
            idxii = idx[id1:id2]
        else:
            idxii = np.arange(id1, id2)
        if id1 > n-1:
            break
        yield idxii
        id1 += bsize

def crossEntropy(yhat, y):
    '''Cross entropy cost function '''
    eps = 1e-10
    yhat = np.clip(yhat, eps, 1-eps)
    aa = y*np.log(yhat)
    return -np.nansum(aa)

def ReLU(x):
    return np.maximum(x, 0)

def dReLU(x):
    '''Gradient of ReLU'''
    return 1.*(x > 0)

def readMNIST(path):
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
