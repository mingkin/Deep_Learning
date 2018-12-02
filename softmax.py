# -*- coding: utf-8 -*-

"""
Author: kingming

File: softmax.py

Time: 2018/12/2 下午6:39

License: (C) Copyright 2018, xxx Corporation Limited.

"""

import numpy as np


def softmax(x):
    '''
    Arguments:
    x -- MxN matrix or vector
    Return
    x -- probability
    '''
    org_shape = x.shape
    print(x.shape)
    if len(x.shape) > 1:
        # Matrix
        print('vector n_dim >1,vector process')
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)

        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator

    else:
        # Vector
        print('vector n_dim<2')
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    assert x.shape == org_shape
    return x


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1.0 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s


def sigmoid_grad(s):
    """
    Arguments:
    s -- A scalar or numpy array.
    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = s * (1 - s)
    ### END YOUR CODE

    return ds




if __name__ == '__main__':
    test1 = np.array([1,2])
    test1_x1 = softmax(test1)

    test2 = np.array([[1, 2], [-1, -2]])
    test2_x2 = softmax(test2)
    test1_x1_1 = sigmoid(test2)
    print(test2_x2,test1_x1_1)












