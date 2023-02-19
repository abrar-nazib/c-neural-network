import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import random

x = list(range(0, 11))
y = []
for i in range(11):
    y.append(random.randint(0, 100))

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)

p = ax.plot(x, y, linewidth="2", color="green")
plt.axis([0, 10, 0, 100])

axslider1 = plt.axes([0.1, 0.2, 0.8, 0.05])
slider1 = Slider(ax=axslider1, label="Slider 1",
                 valmin=0, valmax=100, valinit=30)

axslider2 = plt.axes([0.1, 0.1, 0.8, 0.05])
slider2 = Slider(ax=axslider2, label="Slider 2",
                 valmin=0, valmax=100, valinit=50, valfmt="%1.2f",  color="red")


def update_data(val):
    for i in range(11):
        y[i] += random.randint(-1, 1)*slider2.val
    ax.cla()
    ax.plot(x, y, linewidth="2", color="green")


slider2.on_changed(update_data)


plt.show()
