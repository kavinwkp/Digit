import numpy as np
import tensorflow as tf
from Perceptron import Perceptron
import time
from PIL import Image
import matplotlib.pyplot as plt


class DigitClassifier():

    def __init__(self):
        self.model = Perceptron()
        self.fit()

    def fit(self):
        minst = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = minst.load_data()
        num_train = 15000

        # Train set
        mask = range(num_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        X_train = np.reshape(X_train, (X_train.shape[0], -1))

        # 增加一维
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

        loss_hist = self.model.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                                     num_iters=1500, verbose=True)

        # Loss Profile
        # plt.plot(loss_hist)
        # plt.xlabel('Iteration number')
        # plt.ylabel('Loss value')
        # plt.show()

    def predict(self, x):
        return self.model.predict(x)


def main():
    model = DigitClassifier()
    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()

    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    result = model.predict(X_test)
    print("result: %f" % np.mean(result == y_test), )
    num = [0] * 10
    count = [0] * 10
    num_test = X_test.shape[0]
    for i in range(num_test):
        num[y_test[i]] += 1
        if result[i] == y_test[i]:
            count[y_test[i]] += 1
    print(count)
    print(num)

    print("result: %f" % (np.sum(count) / np.sum(num)))

    accurate = list()
    for i in range(10):
        accurate.append(count[i] / num[i])
    print(np.around(accurate, 3))

# samples: 10000
# [960, 1104, 845, 888, 873, 699, 877, 912, 818, 893]
# [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
# result: 0.886900
# [0.98  0.973 0.819 0.879 0.889 0.784 0.915 0.887 0.84  0.885]


if __name__ == '__main__':
    main()
