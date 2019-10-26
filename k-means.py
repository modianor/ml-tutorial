# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

data_a = 30 * np.random.randn(100, 2)
data_b = 0.6 * np.random.randn(100, 2)

data = np.vstack((data_a, data_b))

print(data)

figure('1')
plt.scatter(data[:, 0], data[:, 1], c="red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

codebook, distortion = kmeans(obs=data, k_or_guess=2)
print(codebook, distortion)

code, dist = vq(obs=data, code_book=codebook)

print(code, dist)

figure('1')

ndx = np.where(code == 0)[0]
plt.plot(data[ndx, 0], data[ndx, 1], '*')

ndx = np.where(code == 1)[0]
plt.plot(data[ndx, 0], data[ndx, 1], 'r.')

plt.plot(codebook[:, 0], codebook[:, 0], 'go')

plt.title('2维数据点聚类')

plt.axis('off')

plt.show()

#
# print(code, dist)

# print(data)
