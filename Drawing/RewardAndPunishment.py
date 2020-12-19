import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


class RewardAndPunishment():

    def __init__(self):
        minst = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = minst.load_data()

        num_train = 4000
        mask = range(num_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])  # 增加一维
        ck, ck_num = np.unique(y_train, return_counts=True)
        ck_idx = list()
        for label in ck:
            label_idx = np.squeeze(np.argwhere(y_train == label))  # 每一类对应的index
            ck_idx.append(label_idx)

        num_train, dim = X_train.shape  # (100, 785)
        num_classes = len(ck)
        self.W = 0.01 * np.random.randn(dim, num_classes)  # 初始化 W

        iterator = 0
        num_iterator = 2000
        while True:
            flag = True
            for n in range(len(ck)):
                X = X_train[ck_idx[n]]  # 取出每一类的所有样本
                for i in range(ck_num[n]):  # 遍历每一个样本
                    score = np.dot(X[i], self.W)
                    if np.argmax(score) != n:
                        flag &= False
                        for j in range(score.shape[0]):
                            if j == n:
                                self.W[:, j] += X[i].T
                            elif score[j] > score[n]:
                                self.W[:, j] -= X[i].T
                    else:
                        flag &= True
            if flag:
                break
            iterator += 1
            if iterator % 100 == 0:
                print('iteration.....%d / %d' % (iterator, num_iterator))
            if iterator > num_iterator:
                break

    def predict(self, x):
        return np.argmax(np.dot(x, self.W))


def main():
    model = RewardAndPunishment()
    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()
    # num_test = 1000
    # mask = range(num_test)
    # X_test = X_test[mask]
    # y_test = y_test[mask]
    # print(X_test.shape)
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

    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])  # 增加一维
    num_class = 10
    num = [0] * num_class
    count = [0] * num_class
    n_test = X_test.shape[0]
    for i in range(n_test):
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

# samples: 1000
# [82, 122, 99, 83, 97, 72, 75, 76, 68, 82]
# [85, 126, 116, 107, 110, 87, 87, 99, 89, 94]
# result: 0.856000
# [0.965 0.968 0.853 0.776 0.882 0.828 0.862 0.768 0.764 0.872]

# samples: 10000
# [929, 1084, 854, 845, 881, 717, 843, 901, 783, 829]
# [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
# result: 0.866600
# [0.948 0.955 0.828 0.837 0.897 0.804 0.88  0.876 0.804 0.822]


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time: ", (end - start))
