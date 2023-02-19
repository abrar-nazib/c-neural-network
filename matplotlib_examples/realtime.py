import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# specify the plot style to be used
plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

# Create the plot and show in the screen
# plt.plot(x_vals, y_vals)

# x = [1, 2, 3]
# y1 = [11, 22, 33]
# y2 = [111, 222, 333]


def animate(i):
    data = pd.read_csv('data.csv')
    # print(data)
    x = data["x_value"]
    y1 = data["total_1"]
    y2 = data["total_2"]

    plt.cla()  # clear the previous plot
    plt.plot(x, y1, marker="x", label="Subscriber Count - Channel 1")
    plt.plot(x, y2, marker="o", label="Subscriber Count - Channel 2")
    # print(x)

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)
# 1. Figure: subplot/figure to animate. plt.gcf() refers to get current figure
# 2. Function: The function to run
# 3. interval: interval in milliseconds

plt.tight_layout()  # For adding some automatic padding to the plot
plt.show()
