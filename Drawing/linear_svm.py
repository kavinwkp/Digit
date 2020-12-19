import numpy as np
from random import shuffle
from past.builtins import xrange
import tensorflow as tf


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)  # 初始化
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)  # (10, )
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:       # 根据公式，j==y[i]的就是本身的分类，不用算了
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1

            if margin > 0:      # 如果计算的margin > 0，那么就要算入loss，
                loss += margin
                dW[:, y[i]] += -X[i, :].T       # 正确项 yi减去X[i]
                dW[:, j] += X[i, :].T           # 正确项 yi减去X[i]

    loss /= num_train
    dW /= num_train

    # 正则项，减小震荡
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    scores = X.dot(W)
    # num_classes = W.shape[1]
    num_train = X.shape[0]
    # 利用np.arange(),correct_class_score变成了 (num_train,y)的矩阵
    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score = np.reshape(correct_class_score, (num_train, -1))

    # sj - syi + delta
    margins = scores - correct_class_score + 1

    # max(0, sj - syi + delta)
    margins = np.maximum(0, margins)

    # 然后这里计算了j=y[i]的情形，所以把他们置为0
    margins[np.arange(num_train), y] = 0
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    margins[margins > 0] = 1
    # 因为j=y[i]的那一个元素的grad要计算 >0 的那些次数次
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum.T
    # 把公式1和2合到一起计算了
    dW = np.dot(X.T, margins)
    dW /= num_train
    dW += reg * W
    return loss, dW
