from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time

class NaiveBayes():
    """朴素贝叶斯算法"""
    def __init__(self):
        minst = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (X_test, y_test) = minst.load_data()
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))

        num_train = 1000
        mask = range(num_train)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]

        # ******************** 二值化 ******************** #
        # for i in range(num_train):
        #     for j in range(784):
        #         if self.X_train[i][j] > 100:
        #             self.X_train[i][j] = 1
        #         else:
        #             self.X_train[i][j] = 0
        # ******************** 二值化 ******************** #

        # ******************** TEST ******************** #
        # self.X_train = np.array([[6, 180, 12], [5.92, 190, 11],
        #                    [5.58, 170, 12], [5.92, 165, 10],
        #                    [5, 100, 6], [5.5, 150, 8],
        #                    [5.42, 130, 7], [5.75, 150, 9]])
        # self.y_train = np.array(([1, 1, 1, 1, 0, 0, 0, 0]))
        # num_train = self.X_train.shape[0]
        # ******************** TEST ******************** #

        self.ck, self.ck_num = np.unique(self.y_train, return_counts=True)
        # print(ck)   # [0 1 2 3 4 5 6 7 8 9]
        # print(ck_num)     # [13 14  6 11 11  5 11 10  8 11]

        self.x = None
        self.pw = np.zeros((len(self.ck),))    # 先验概率

        for i in range(len(self.ck)):
            self.pw[i] = self.ck_num[i] / num_train
        # print(pw)

        self.ck_idx = list()
        for label in self.ck:
            label_idx = np.squeeze(np.argwhere(self.y_train == label))   # 每一类对应的index
            self.ck_idx.append(label_idx)
        # print(ck_idx)

    def predict(self, X):
        S = np.zeros((self.X_train.shape[1], self.X_train.shape[1]))  # (784, 784)
        hx = np.zeros((len(self.ck),))  # 后验概率
        for index in range(len(self.ck)):
            x = X.astype(np.float64)
            # print("calculating... %d" % (index))
            X_data = self.X_train[self.ck_idx[index]]     # 取出每一类的所有样本
            X_data = X_data.astype(np.float64)

            X_means = np.sum(X_data, axis=0)
            X_means = X_means / X_data.shape[0]

            for i in range(self.X_train.shape[1]):       # 784
                for j in range(self.X_train.shape[1]):   # 784
                    S[i][j] = np.dot((X_data[:, i] - X_means[i]), X_data[:, j] - X_means[j]) / (X_data.shape[0] - 1)

            S_ = np.linalg.pinv(S)
            det = np.linalg.det(S)

            x = x - X_means
            t = np.dot(x.transpose(), S_)
            t1 = np.dot(t, x)
            t2 = np.log(self.pw[index])
            t3 = np.log(det + 1)
            hx[index] = -t1 + t2 * 2 - t3
        # print("probably: ", hx)
        return np.argmax(hx, axis=0)


def main():
    model = NaiveBayes()
    # ******************** TEST ******************** #
    # X_test = np.array([[6, 130, 8], [5.42, 130, 7]])
    # y_test = np.array(([1, 0]))
    # ******************** TEST ******************** #
    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()
    num_test = 10
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # ******************** 二值化 ******************** #
    # for i in range(num_test):
    #     for j in range(784):
    #         if X_test[i][j] > 100:
    #             X_test[i][j] = 1
    #         else:
    #             X_test[i][j] = 0
    # ******************** 二值化 ******************** #

    # fig = plt.figure(figsize=(6, 6))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # # 绘制数字：每张图像8*8像素点
    # for i in range(num_test):
    #     ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    #     ax.imshow(X_test[i].reshape(28, 28), cmap='binary', interpolation='nearest')
    #     # 用目标值标记图像
    #     ax.text(0, 7, str(y_test[i]))
    # plt.show()

    ck = 10
    num = [0] * ck
    count = [0] * ck
    n_test = X_test.shape[0]
    for i in range(n_test):
        print("predict: ", i)
        y_pred = model.predict(X_test[i])
        num[y_test[i]] += 1
        if y_pred == y_test[i]:
            count[y_test[i]] += 1

    print(count)    # 正确识别数
    print(num)      # 测试样本数

    # print("result: %f" % (np.sum(count) / np.sum(num)))       # 总正确率
    #
    # accurate = list()
    # for i in range(10):
    #     accurate.append(count[i] / num[i])
    # print(np.around(accurate, 3))         # 每类正确率， 保留3位小数

    # samples: 100
    # result: 0.480000
    # [1.    0.    0.875 0.818 0.643 0.429 0.6   0.133 1.    0.182]

    # samples: 300
    # result: 0.553333
    # [1.    0.    0.938 0.833 0.703 0.724 0.667 0.088 0.81  0.265]

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time: ", (end - start))