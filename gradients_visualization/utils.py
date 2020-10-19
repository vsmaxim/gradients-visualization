from typing import Callable

import torch


def grad(torch_f: Callable, x: float, y: float):
    x_t = torch.tensor([x], requires_grad=True)
    y_t = torch.tensor([y], requires_grad=True)
    res = torch_f(x_t, y_t)
    res.backward()
    return x_t.grad[0].item(), y_t.grad[0].item()
