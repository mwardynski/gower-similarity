import random

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def metric(vec_1, vec_2):
    print(f"{vec_1[0]} - {type(vec_1[0])}\n{vec_2} - {type(vec_2)}")
    return np.abs(np.sum(vec_1 - vec_2))


if __name__ == "__main__":
    # data = np.array([[1, 2, 3],
    #                  [2, 1, 3],
    #                  [3, 0, 0],
    #                  [1, 3, 1],
    #                  [5, 6, 2],
    #                  [7, 5, 6],
    #                  [9, 0, 0]])
    #
    # data_y = np.array([0, 1, 0, 1, 0, 0, 1])
    #
    # test = np.array([[5, 4, 3],
    #                  [2, 1, 0]])
    # test_y = np.array([0, 1])
    #
    # knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    # knn.fit(data, data_y)
    # print(knn.score(test, test_y))

    a = np.array([1, 5, 3, 9, -1, 10, 11, 5, 7, 8, 4], dtype=np.int32)
    print(np.ptp(a))
