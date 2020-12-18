import numpy as np

X_train = np.array([[6, 180, 12], [5.92, 190, 11],
                    [5.58, 170, 12], [5.92, 165, 10],
                    [5, 100, 6], [5.5, 150, 8],
                    [5.42, 130, 7], [5.75, 150, 9]])



y_train = np.array(([1, 1, 1, 1, 0, 0, 0, 0]))
# print(X_train.shape)    # (8, 3)
# print(y_train.shape)    # (8,)
# print(X_train.dtype)    # float64
num_train = X_train.shape[0]
ck, ck_num = np.unique(y_train, return_counts=True)
# print(ck)       # [0 1]
# print(ck_num)   # [4 4]

pw = np.zeros((len(ck),))    # 先验概率
hx = np.zeros((len(ck),))    # 后验概率
for i in range(len(ck)):
    pw[i] = ck_num[i] / num_train
# print(pw)     # [0.5 0.5]

ck_idx = list()
for label in ck:
    label_idx = np.squeeze(np.argwhere(y_train == label))   # 每一类对应的index
    ck_idx.append(label_idx)
# print(ck_idx)       # [0, 1] --> [array([4, 5, 6, 7]), array([0, 1, 2, 3])]

X_means = np.zeros((X_train.shape[1], ))       # (3, )
S = np.zeros((X_train.shape[1], X_train.shape[1]))          # (3, 3)

for index in range(2):
    x = np.array([6, 130, 8])
    print("calculating... %d" % (index))
    X_data = X_train[ck_idx[index]]     # 取出每一类的所有样本
    X_data = X_data.astype(np.float64)

    X_means = np.sum(X_data, axis=0)
    X_means = X_means / X_data.shape[0]
    print(X_means)
    # plt.imshow(X_means.reshape(28, 28), cmap="binary")

    for i in range(X_train.shape[1]):       # 784
        for j in range(X_train.shape[1]):   # 784
            S[i][j] = np.dot((X_data[:, i] - X_means[i]), X_data[:, j] - X_means[j]) / (X_data.shape[0] - 1)

    # print(S)
    # u, s, vh = np.linalg.svd(S)
    # print('分解得到矩阵的形状：\n', u.shape, s.shape, vh.shape)
    # print(u)
    # print(vh)
    # print('奇异值：\n', s)
    print(S)
    S_ = np.linalg.inv(S)
    print(S_)
    det = np.linalg.det(S)
    print("det: ", det)
    # print(det)
    print('样本：', x)
    x = x - X_means
    print("x - xmeans: ", x)
    t = np.dot(x.transpose(), S_)
    print(t)
    t1 = np.dot(t, x)
    print("t1: ", t1)
    # t2 = np.log(pw[index])
    t3 = np.log(det)
    # hx[index] = -t1 + t2 * 2 - t3
    hx[index] = -0.5 * (t3 + t1 + 3 * np.log(2 * np.pi))
print(hx)


# count = np.array([8, 0, 7, 9, 9, 3, 6, 2, 2, 2])
# num = np.array([8, 14, 8, 11, 14, 7, 10, 15, 2, 11])
count = np.array([24, 0, 30, 20, 26, 21, 16, 3, 17, 9])
num = np.array([24, 41, 32, 24, 37, 29, 24, 34, 21, 34])
print(count)  # 正确识别数
print(num)  # 测试样本数

print("result: %f" % (np.sum(count) / np.sum(num)))       # 总正确率

accurate = list()
for i in range(10):
    accurate.append(count[i] / num[i])
print(np.around(accurate, 3))         # 每类正确率， 保留3位小数

# samples: 100
# result: 0.480000
# [1.    0.    0.875 0.818 0.643 0.429 0.6   0.133 1.    0.182]

# samples: 300
# result: 0.553333
# [1.    0.    0.938 0.833 0.703 0.724 0.667 0.088 0.81  0.265]
