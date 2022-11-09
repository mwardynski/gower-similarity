import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def metric(vec_1, vec_2):
    print(f"{vec_1[0]} - {type(vec_1[0])}\n{vec_2} - {type(vec_2)}")
    return np.abs(np.sum(vec_1 - vec_2))


if __name__ == "__main__":

    # results = np.zeros((13, 7, 2))
    #
    # with open("hierarchical_clustering_res.txt", "r") as f:
    #     lanes = f.readlines()
    #     for i in range(len(lanes)):
    #         lanes[i] = lanes[i].split(" ")
    #
    #     index_1 = 0
    #     index_2 = 0
    #     for e, lane in enumerate(lanes):
    #         if index_1 == 7 or e == 0:
    #             if e != 0:
    #                 index_1 = 0
    #                 index_2 += 1
    #             continue
    #
    #         results[index_2][index_1][0], results[index_2][index_1][1] = (
    #             lane[0],
    #             lane[1],
    #         )
    #         index_1 += 1
    #
    #     X = [50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250]
    #     Y_cpcc = [[results[j][i][0] for j in range(13)] for i in range(7)]
    #     Y_ioa = [[results[j][i][1] for j in range(13)] for i in range(7)]
    #     labels = {
    #         0: "gower",
    #         1: "bin",
    #         2: "euclidean",
    #         3: "cosine",
    #         4: "minkowski",
    #         5: "dice",
    #         6: "jaccard",
    #     }
    #
    #     # Plot cpcc results
    #     for i in range(7):
    #         plt.plot(X, Y_cpcc[i], label=labels[i])
    #     plt.legend()
    #     plt.title("CPCC")
    #     plt.show()
    #
    #     for i in range(7):
    #         plt.plot(X, Y_ioa[i], label=labels[i])
    #     plt.legend()
    #     plt.title("IoA")
    #     plt.show()

    A = np.array([[1 for _ in range(5)] for _ in range(5)])
    indices_row, indices_col = np.triu_indices(5, k=1)
    print(indices_row, indices_col)
    A[indices_row, indices_col] = 2
    print(A)
