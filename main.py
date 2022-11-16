import random

from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize

from enum import Enum

import timeit


class SimpleEnum(Enum):
    example_0 = 0
    example_1 = 1
    example_2 = 2


if __name__ == "__main__":
    a = np.array([SimpleEnum.example_0 for _ in range(1000)])
    b = np.array([0 for _ in range(1000)])

    start = timeit.default_timer()
    eq_1 = a == SimpleEnum.example_0
    print("{:.2e}".format(timeit.default_timer() - start))

    start = timeit.default_timer()
    eq_2 = b == 0
    print("{:.2e}".format(timeit.default_timer() - start))

    start = timeit.default_timer()
    eq_3 = (a == SimpleEnum.example_0) | (b == 0)
    print("{:.2e}".format(timeit.default_timer() - start))
