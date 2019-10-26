import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, decomposition


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def test_KPCA(*data):
    X, Y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components=None, kernel=kernel)
        kpca.fit(X)
        print("kernel=%s-->lambdas:%s" % (kernel, kpca.lambdas_))


def plot_KPCA(*data):
    X, Y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()
    colors = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
    (0, 0.6, 0.4), (0.5, 0.3, 0.2),)
    for i, kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components=2, kernel=kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2, 2, i + 1)
        for label, color in zip(np.unique(Y), colors):
            position = Y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label="target=%d" % label, color=color)
            ax.set_xlabel("X[0]")
            ax.set_ylabel("X[1]")
            ax.legend(loc="best")
            ax.set_title("kernel=%s" % kernel)
    plt.suptitle("KPCA")
    plt.show()


X, Y = load_data()
test_KPCA(X, Y)
plot_KPCA(X, Y)
