import numpy as np
from past.builtins import xrange
from linear_svm import svm_loss_vectorized
from linear_svm import svm_loss_naive


class Perceptron():
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)      # 初始化 W

        loss_history = []
        for it in xrange(num_iters):
            batch_inx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_inx, :]
            y_batch = y[batch_inx]

            loss, grad = self.loss(X_batch, y_batch, reg)   # 计算损失和梯度
            loss_history.append(loss)

            self.W = self.W - learning_rate * grad      # 更新 W

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        score = X.dot(self.W)
        y_pred = np.argmax(score, axis=1)

        return y_pred

    def loss(self, X, y, reg):
        dW = np.zeros(self.W.shape)  # 初始化
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in xrange(num_train):
            scores = X[i].dot(self.W)  # (10, )
            correct_class_score = scores[y[i]]
            for j in xrange(num_classes):
                if j == y[i]:       # j==y[i]不用算
                    continue
                margin = scores[j] - correct_class_score + 1  # delta = 1

                if margin > 0:  # 如果计算的margin > 0，那么就要算入loss，
                    loss += margin
                    dW[:, y[i]] += -X[i, :].T   # 正确项 yi 减去X[i]
                    dW[:, j] += X[i, :].T       # 错误项 j  加上X[i]

        loss /= num_train
        dW /= num_train

        # 正则项，减小震荡
        loss += reg * np.sum(self.W * self.W)
        dW += reg * self.W

        return loss, dW