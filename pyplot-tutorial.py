import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 100, 30)
y = 3 * x + 7 + np.random.randn(30) * 6

plt.scatter(x, y)

plt.show()
