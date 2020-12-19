import numpy as np
import tensorflow as tf
from past.builtins import xrange
import matplotlib.pyplot as plt
import time
class Fisher():
    def __init__(self):
        minst = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (X_test, y_test) = minst.load_data()
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))

        num_train = 1000
        mask = range(num_train)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]

        self.ck, self.ck_num = np.unique(self.y_train, return_counts=True)
        # print(ck_num)       # [13 14  6 11 11  5 11 10  8 11]

        self.ck_idx = list()
        for label in self.ck:
            label_idx = np.squeeze(np.argwhere(self.y_train == label))   # 每一类对应的index
            self.ck_idx.append(label_idx)

        # plt.imshow(X_data.reshape(28, 28), cmap="binary")
        # plt.show()

    def predict(self, X):
        classes = np.zeros((len(self.ck), len(self.ck)), dtype=int)
        pnum = np.zeros((len(self.ck), ))
        for m in range(len(self.ck)):
            print("calculating... %d" % (m))
            sample1 = self.X_train[self.ck_idx[m]]  # 取出每一类的所有样本
            sample1 = sample1.astype(np.float64)
            num1 = self.ck_num[m]       # 样品个数
            M1 = np.sum(sample1, axis=0) / num1       # 按列求和

            # 类内离散度矩阵
            S1 = np.dot((sample1 - M1).transpose(), sample1 - M1) / (num1 - 1)

            for n in range(len(self.ck)):
                sample2 = self.X_train[self.ck_idx[n]]  # 取出每一类的所有样本
                sample2 = sample2.astype(np.float64)
                num2 = self.ck_num[n]
                M2 = np.sum(sample2, axis=0) / num2
                S2 = np.dot((sample2 - M2).transpose(), sample2 - M2) / (num2 - 1)
                Sw = S1 + S2        # 总类内离散度矩阵
                Sw_ = np.linalg.pinv(Sw)
                M = M1 - M2
                w = np.dot(M, Sw_)      # 向量 w*
                m1 = np.sum(np.dot(w, sample1.transpose())) / num1      # 投影求均值
                m2 = np.sum(np.dot(w, sample2.transpose())) / num2
                # y0 = 0.5 * (m1 + m2)
                y0 = (num1 * m1 + num2 * m2) / (num1 + num2)
                y = np.dot(w, X.transpose())
                if y > y0:
                    classes[m][n] = m
                else:
                    classes[m][n] = n
                pnum[classes[m][n]] += 1
        print(pnum)
        return np.argmax(pnum, axis=0)


def main():
    model = Fisher()
    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()
    num_test = 100
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

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

    print("result: %f" % (np.sum(count) / np.sum(num)))       # 总正确率

    accurate = list()
    for i in range(10):
        accurate.append(count[i] / num[i])
    print(np.around(accurate, 3))         # 每类正确率， 保留3位小数

# samples: 100
# [8, 13, 6, 9, 14, 6, 8, 9, 2, 7]
# [8, 14, 8, 11, 14, 7, 10, 15, 2, 11]
# result: 0.820000
# [1.    0.929 0.75  0.818 1.    0.857 0.8   0.6   1.    0.636]


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time: ", (end - start))