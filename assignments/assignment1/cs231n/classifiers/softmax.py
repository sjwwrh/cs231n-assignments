from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # for gradient computation
    L1 = np.zeros(shape=(num_train, num_classes))
    L3 = np.zeros_like(L1)
    sum = np.zeros(shape=(num_train, 1))
    L4 = np.zeros_like(L1)

    for i in range(num_train):
        scores = X[i].dot(W) #L1
        L1[i] = scores

        # compute the probabilities in numerically stable way
        scores -= np.max(scores) #L2
        p = np.exp(scores) #L3
        L3[i] = p
        sum[i] = p.sum()
        p /= p.sum()  # normalize L4
        L4[i] = p
        logp = np.log(p) #L5

        loss -= logp[y[i]]  # negative log probability is the loss #L6


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    dL6 = -1 * np.ones(shape=(num_train, num_classes))
    dL5 = np.zeros_like(dL6)
    for i in range(num_train):
        dL5[i, y[i]] = 1
    dL5 = np.multiply(dL5, dL6)
    dL4 = np.divide(dL5, L4)
    dL3 = np.zeros_like(dL4)
    for i in range(num_train):
        for j in range(num_classes):
            if j == y[i]:
                dL3[i, j] = (sum[i] - L3[i, j])
            else:
                dL3[i, j] = -L3[i, y[i]]
        dL3[i] = dL4[i, y[i]] * dL3[i] / sum[i]**2
    
    dL2 = np.multiply(dL3, L3)
    dW = np.matmul(X.T, dL2) / num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X @ W
    scores -= np.max(scores, axis=1).reshape(-1, 1)
    p = np.exp(scores)
    p /= np.sum(p, axis=1).reshape(-1, 1)
    logp = np.log(p)
    loss -= np.sum(logp[np.arange(num_train), y])
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    one_hot = np.zeros(shape=(num_train, num_classes))
    one_hot[np.arange(num_train), y] = 1
    dscores = p - one_hot
    dW = np.matmul(X.T, dscores) / num_train
    dW += 2 * reg * W

    return loss, dW
