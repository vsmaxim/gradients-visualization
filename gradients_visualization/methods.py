from typing import Callable

import numpy as np

from gradients_visualization.utils import grad


def build_vanilla_gd(f: Callable, torch_f: Callable, lr: float = 0.01):

    def vanilla_gd(x: np.array, y: np.array, z: np.array):
        last_x, last_y = x[-1], y[-1]
        x_grad, y_grad = grad(torch_f, last_x, last_y)
        new_x, new_y = last_x - lr * x_grad, last_x - lr * y_grad
        return np.append(x, new_x), np.append(y, new_y), np.append(z, f(new_x, new_y))

    return vanilla_gd


def build_momentum_gd(f: Callable, torch_f: Callable, lr: float = 0.01, decay: float = 0.8):
    storage = {"x_sum": 0, "y_sum": 0}

    def momentum_gd(x: np.array, y: np.array, z: np.array):
        last_x, last_y = x[-1], y[-1]
        x_grad, y_grad = grad(torch_f, last_x, last_y)

        storage["x_sum"] = x_grad + storage["x_sum"] * decay
        storage["y_sum"] = y_grad + storage["y_sum"] * decay

        delta_x = - lr * storage["x_sum"]
        delta_y = - lr * storage["y_sum"]

        new_x = last_x + delta_x
        new_y = last_y + delta_y

        return np.append(x, new_x), np.append(y, new_y), np.append(z, f(new_x, new_y))

    return momentum_gd


def build_adagrad(f: Callable, torch_f: Callable, lr: float = 0.1):
    storage = {"x_sum": 0, "y_sum": 0}

    def adagrad(x: np.array, y: np.array, z: np.array):
        last_x, last_y = x[-1], y[-1]
        x_grad, y_grad = grad(torch_f, last_x, last_y)

        storage["x_sum"] += x_grad ** 2
        storage["y_sum"] += y_grad ** 2

        delta_x = - lr * x_grad / np.sqrt(storage["x_sum"])
        delta_y = - lr * y_grad / np.sqrt(storage["y_sum"])

        new_x = last_x + delta_x
        new_y = last_y + delta_y

        return np.append(x, new_x), np.append(y, new_y), np.append(z, f(new_x, new_y))

    return adagrad


def build_rmsprop(f: Callable, torch_f: Callable, lr: float = 0.01, decay: float = 0.99):
    storage = {"x_sum": 0, "y_sum": 0}

    def rmsprop(x: np.array, y: np.array, z: np.array):
        last_x, last_y = x[-1], y[-1]
        x_grad, y_grad = grad(torch_f, last_x, last_y)

        storage["x_sum"] = storage["x_sum"] * decay + x_grad ** 2 * (1 - decay)
        storage["y_sum"] = storage["y_sum"] * decay + y_grad ** 2 * (1 - decay)

        delta_x = - lr * x_grad / np.sqrt(storage["x_sum"])
        delta_y = - lr * y_grad / np.sqrt(storage["y_sum"])

        new_x = last_x + delta_x
        new_y = last_y + delta_y
        return np.append(x, new_x), np.append(y, new_y), np.append(z, f(new_x, new_y))

    return rmsprop


def build_adam(f: Callable, torch_f: Callable, lr: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999):
    storage = {"x_sum": 0, "y_sum": 0, "x_2_sum": 0, "y_2_sum": 0}

    def adam(x: np.array, y: np.array, z: np.array):
        last_x, last_y = x[-1], y[-1]
        x_grad, y_grad = grad(torch_f, last_x, last_y)

        storage["x_sum"] = storage["x_sum"] * beta_1 + x_grad * (1 - beta_1)
        storage["y_sum"] = storage["y_sum"] * beta_1 + y_grad * (1 - beta_1)

        storage["x_2_sum"] = storage["x_2_sum"] * beta_2 + x_grad ** 2 * (1 - beta_2)
        storage["y_2_sum"] = storage["y_2_sum"] * beta_2 + y_grad ** 2 * (1 - beta_2)

        delta_x = - lr * storage["x_sum"] / np.sqrt(storage["x_2_sum"])
        delta_y = - lr * storage["y_sum"] / np.sqrt(storage["y_2_sum"])

        new_x = last_x + delta_x
        new_y = last_y + delta_y
        return np.append(x, new_x), np.append(y, new_y), np.append(z, f(new_x, new_y))

    return adam
