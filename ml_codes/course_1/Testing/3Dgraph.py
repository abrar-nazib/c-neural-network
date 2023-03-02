import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Single points in 3D
ax = plt.axes(projection="3d")
# xdata = np.random.randint(low=-100, high=100, size=(500,))
# ydata = np.random.randint(low=-100, high=100, size=(500,))
# zdata = np.random.randint(low=0, high=100, size=(500, ))
# ax.scatter(xdata, ydata, zdata, marker="v")

xdata = np.arange(-5, 5, 0.05)
ydata = np.arange(-5, 5, 0.05)

X, Y = np.meshgrid(xdata, ydata)

# print(xdata)
# print("________________")
# print(X)
# zdata = np.sin(xdata) * np.sin(ydata)

Z = np.sin(X)+np.cos(Y)


ax.plot_surface(X, Y, Z, cmap="plasma")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
