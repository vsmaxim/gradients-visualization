from collections import Callable
import random
from typing import Tuple

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def visualize(
    method: Callable,
    func: Callable,
    x_interval: Tuple[float, float],
    y_interval: Tuple[float, float]
):
    def updater(update_func, start_point=4 * np.random.rand(2) - 2):
        start_x_val, start_y_val = start_point
        xx = np.array([start_x_val])
        yy = np.array([start_y_val])
        zz = np.array([func(start_x_val, start_y_val)])

        opt_line.set_data(xx, yy)
        opt_line.set_3d_properties(zz)

        def update(frame):
            xx, yy, zz = update_func(*opt_line.get_data_3d())

            # Bounds
            xx[xx < x_left] = x_left
            xx[xx > x_right] = x_right
            yy[yy < y_left] = y_left
            yy[yy > y_right] = y_right

            # Trail remove
            if len(xx) > 5:
                xx = xx[1:]
                yy = yy[1:]
                zz = zz[1:]

            opt_line.set_data(xx, yy)
            opt_line.set_3d_properties(zz)
            return opt_line,

        return update

    def init():
        return opt_line,

    x = np.linspace(*x_interval, 100)
    y = np.linspace(*y_interval, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    opt_line, = ax.plot([], [], [], "ro")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    x_left, x_right = x_interval
    y_left, y_right = y_interval

    start_x = float(random.randint(*map(int, x_interval)))
    start_y = float(random.randint(*map(int, y_interval)))

    a = FuncAnimation(
        fig,
        updater(method, start_point=(start_x, start_y)),
        frames=1000,
        init_func=init,
        blit=True
    )

    plt.show()
