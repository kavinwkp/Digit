from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

print(X_train[0:10])
print(y_train[0:10])

