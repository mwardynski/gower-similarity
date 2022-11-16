import random

from time import sleep

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize


def metric(vec_1, vec_2):
    print(f"{vec_1[0]} - {type(vec_1[0])}\n{vec_2} - {type(vec_2)}")
    return np.abs(np.sum(vec_1 - vec_2))


def func(w):
    print("func", w[0] ** 2 + w[1] ** 2)
    return w[0] ** 2 + w[1] ** 2


def derivative(w):
    print("deriv", np.array([2 * w[0] + w[1] ** 2, w[0] ** 2 + 2 * w[1] ** 0]))
    return np.array([2 * w[0] + w[1] ** 2, w[0] ** 2 + 2 * w[1] ** 0])


if __name__ == "__main__":
    a = np.array([1.0])
    b = np.array([2.0])
    print(a, b)

    c = a > b
    print(b[~c])
