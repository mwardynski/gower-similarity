import random

import numpy as np
import matplotlib.pyplot as plt

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
    results = np.zeros((13, 7, 2))

    with open("hierarchical_clustering_res.txt", "r") as f:
        lanes = f.readlines()
        for i in range(len(lanes)):
            lanes[i] = lanes[i].split(" ")

        index_1 = 0
        index_2 = 0
        for e, lane in enumerate(lanes):
            if index_1 == 7 or e == 0:
                if e != 0:
                    index_1 = 0
                    index_2 += 1
                continue

            results[index_2][index_1][0], results[index_2][index_1][1] = (
                lane[0],
                lane[1],
            )
            index_1 += 1

        X = [50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250]
        Y_cpcc = [[results[j][i][0] for j in range(13)] for i in range(7)]
        Y_ioa = [[results[j][i][1] for j in range(13)] for i in range(7)]
        labels = {
            0: "gower",
            1: "bin",
            2: "euclidean",
            3: "cosine",
            4: "minkowski",
            5: "dice",
            6: "jaccard",
        }

        # Plot cpcc results
        for i in range(7):
            plt.plot(X, Y_cpcc[i], label=labels[i])
        plt.legend()
        plt.title("CPCC")
        plt.show()

        for i in range(7):
            plt.plot(X, Y_ioa[i], label=labels[i])
        plt.legend()
        plt.title("IoA")
        plt.show()
