import os.path
from os import listdir
from os.path import isfile
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils.validation
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, single
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

import CopheneticTree


class DataType(Enum):
    BINARY_SYMMETRIC = 0
    BINARY_ASYMMETRIC = 1
    CATEGORICAL_NOMINAL = 2
    RATIO_SCALE = 3


@dataclass
class Dataset:
    name: str
    task: str
    metric: str


@dataclass
class Data:
    cols_type_maping = {
        "bin_sym": DataType.BINARY_SYMMETRIC,
        "bin_asym": DataType.BINARY_ASYMMETRIC,
        "cat_nom": DataType.CATEGORICAL_NOMINAL,
        "ratio": DataType.RATIO_SCALE,
        "NULL": None,
    }
    data: dict
    cols_type: dict
    labels: dict


class GowerMetric:
    def __init__(self, dtypes: np.array, ratio_scale_normalization="range"):
        self.dtypes = dtypes  # initialize with np.array of column data types
        self.ranges_: List  # values of ranges in ._ratio_scale() (iqr or traditional range)
        self.h_: List  # h values in ._ratio_scale()
        self.n_features_in_: int
        self._sigma_sum = 0  # counter for existing values in vector
        self.ratio_scale_normalization: str = ratio_scale_normalization
        self._similarities_map = {
            DataType.BINARY_SYMMETRIC: self._bin_sym,
            DataType.BINARY_ASYMMETRIC: self._bin_asym,
            DataType.CATEGORICAL_NOMINAL: self._cat_nom,
            DataType.RATIO_SCALE: self._ratio_scale,
        }

    def __call__(
        self,
        vector_1: np.array,
        vector_2: np.array,
    ):
        sklearn.utils.validation.check_is_fitted(self)

        if self.n_features_in_ != len(vector_1) | len(vector_2):
            print("Vector sizes don't match!")
            print(self.n_features_in_, len(vector_1), len(vector_2))
            return -1

        dist_func = lambda similarity, a, b: 1 - similarity(a, b)
        dist_func_ratio = lambda similarity, a, b, r, h: 1 - similarity(
            a, b, r, h
        )
        distances = [
            dist_func(
                self._similarities_map[self.dtypes[i]],
                vector_1[i],
                vector_2[i],
            )
            if self.dtypes[i] != DataType.RATIO_SCALE
            else dist_func_ratio(
                self._similarities_map[self.dtypes[i]],
                vector_1[i],
                vector_2[i],
                self.ranges_[i],
                self.h_[i],
            )
            for i in range(self.n_features_in_)
        ]
        distance = np.sum(distances)

        distance /= self._sigma_sum
        self._reset_sigma()

        return distance

    def fit(self, X):
        self.ranges_ = [0 for _ in range(self.dtypes.size)]
        self.h_ = [0 for _ in range(self.dtypes.size)]
        self.n_features_in_ = self.dtypes.size

        for i in range(self.dtypes.size):
            if self.dtypes[i] == DataType.RATIO_SCALE:
                column = X[:, i]

                if self.ratio_scale_normalization == "iqr":
                    # IQR (g_t) - Interquartile Range
                    q1, q3 = np.percentile(column, [25, 75])
                    self.ranges_[i] = q3 - q1
                elif self.ratio_scale_normalization == "range":
                    # Traditional range of values
                    self.ranges_[i] = np.ptp(column, axis=0)
                else:
                    print(
                        "GowerMetric - Ratio scale has wrong type of normalization!"
                    )
                    raise ValueError

                # h_t - bandwidth in the kernel density estimation (Marcello D’Orazio - p. 9)
                c = 1.06

                s = np.std(column)
                n = X[:, i].size
                self.h_[i] = (
                    c / n ** (1 / 5) * np.min([s, self.ranges_[i] / 1.34])
                )
        # print('Gower\'s Metric fit')
        # print(f'IQRS: {self.ranges_}')
        # print(f'h: {self.h_}')

    def _bin_sym(self, value_1, value_2) -> float:
        self._add_sigma()
        return 1.0 if np.isclose(value_1, value_2) else 0.0

    def _bin_asym(self, value_1, value_2) -> float:
        if np.isclose(value_1, value_2) and np.isclose(value_1, 1.0):
            self._add_sigma()
            return 1.0
        else:
            return 0.0

    def _cat_nom(self, value_1, value_2) -> float:
        self._add_sigma()
        return 1.0 if np.isclose(value_1, value_2) else 0.0

    def _ratio_scale(self, value_1, value_2, val_range, h) -> float:
        self._add_sigma()
        absolute = np.abs(value_1 - value_2)
        if absolute >= val_range:
            return 0.0
        elif absolute <= h:
            return 1.0
        else:
            return 1.0 - absolute / val_range

    def _add_sigma(self):
        self._sigma_sum += 1

    def _reset_sigma(self):
        self._sigma_sum = 0


def bin_dist(vector_1: np.ndarray, vector_2: np.ndarray):
    # bool array for not-null values
    non_null_map_1 = vector_1 != -1
    non_null_map_2 = vector_2 != -1
    non_null_map = non_null_map_1 & non_null_map_2

    # vec_1 and vec_2 with only not-null values
    non_null_1 = vector_1[non_null_map]
    non_null_2 = vector_2[non_null_map]

    # count null values
    null_count = (~non_null_map_1 | ~non_null_map_2).sum()

    # return sum of dissimilarities between vec_1 and vec_2 + number of unique null fields
    return ((~np.isclose(non_null_1, non_null_2)).sum() + null_count) / len(
        vector_1
    )


# Simple function for calculating dissimilarity matrix
def make_matrix(data_frame: np.ndarray, metric) -> np.array:
    return pdist(data_frame, metric)


def cpcc(x, t):

    x_row, x_col = np.triu_indices(x.shape[0], k=1)
    t_row, t_col = np.triu_indices(t.shape[0], k=1)

    x_ = np.average(x[x_row, x_col])
    t_ = np.average(t[t_row, t_col])

    return np.sum((x[x_row, x_col] - x_) * (t[t_row, t_col] - t_)) / np.sqrt(
        np.sum(np.power(x[x_row, x_col] - x_, 2))
        * np.sum(np.power(t[t_row, t_col] - t_, 2))
    )


def ioa(O, P):
    # Z = single(pdist(X, metric=metric_func))
    # P = np.triu(squareform(cophenet(Z)))

    O_row, O_col = np.triu_indices(O.shape[0], k=1)
    P_row, P_col = np.triu_indices(P.shape[0], k=1)

    O_ = np.average(O[O_row, O_col])
    return 1 - np.sum(np.power(P[P_row, P_col] - O[O_row, O_col], 2)) / np.sum(
        np.power(
            np.absolute(P[P_row, P_col] - O_)
            + np.absolute(O[O_row, O_col] - O_),
            2,
        )
    )


def knn_test(dataset: Dataset, data: Data, number_of_records: int = None):
    if number_of_records is None:
        number_of_records = len(data.data[dataset.name])

    df = np.copy(data.data[dataset.name][:number_of_records])
    # print('Data:', df)
    df = fill_na(df)

    gower = GowerMetric(data.cols_type[dataset.name], "iqr")

    if dataset.metric == "gower":
        metric_func = gower
    elif dataset.metric == "bin":
        metric_func = bin_dist
    else:
        metric_func = dataset.metric

    print(
        f"----------------- Test using {dataset.metric} metric -----------------"
    )
    if dataset.task == "bin" or dataset.task == "multivar":
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        cat_nom_cols = [
            i
            for i in range(len(gower.dtypes))
            if gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
        ] + [len(gower.dtypes)]
        fit_df = df[:, cat_nom_cols]

        enc.fit(fit_df)
        fit_df = enc.transform(fit_df)
        df[:, cat_nom_cols] = fit_df

        y = df[:, -1]
        df = df[:, :-1]

        df = np.ndarray.astype(df, dtype=np.float64)
        y = np.ndarray.astype(y, dtype=np.float64)

        train_set, test_set, y_train_set, y_test_set = train_test_split(
            df, y, test_size=0.2
        )

        if dataset.metric == "gower":
            gower.fit(train_set)

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric_func)
        knn.fit(train_set, y_train_set)

        score = knn.score(test_set, y_test_set)

        print(f"KNN score: {score}")
        return score

    elif dataset.task == "reg":
        train_set, test_set = train_test_split(df, test_size=0.3)

        gower.fit(train_set)

        knn = NearestNeighbors(n_neighbors=5, metric=metric_func)
        knn.fit(train_set)
        print(knn.kneighbors(test_set, 5, False))

    elif dataset.task == "cluster":
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        # Categorical Nominal columns
        cat_nom_cols = [
            i
            for i in range(len(gower.dtypes))
            if gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
        ] + [len(gower.dtypes)]
        fit_df = df[:, cat_nom_cols]

        enc.fit(fit_df)
        fit_df = enc.transform(fit_df)
        df[:, cat_nom_cols] = fit_df

        df = df[:, :-1]

        df = np.ndarray.astype(df, dtype=np.float64)

        if dataset.metric == "gower":
            gower.fit(df)

        # Hierarchical Clustering and dendrogram (without plotting)
        Z = linkage(df, method="average", metric=metric_func)

        # plt.figure()

        dn = dendrogram(Z, no_plot=True)

        # distances between elements using given metric function
        dist_x = np.zeros((number_of_records, number_of_records))
        row, col = np.triu_indices(number_of_records, 1)
        dist_x[row, col] = pdist(df, metric=metric_func)

        dist_t = CopheneticTree.get_cophenetic_distance_matrix(
            dn["dcoord"], dn["leaves"]
        )

        c = cpcc(dist_x, dist_t)
        i = ioa(dist_x, dist_t)
        print(f"CPCC: {c}")
        print(f"IoA: {i}")
        return c, i

    else:
        print("Wrong task!")
    print(
        "--------------------------------------------------------------------\n"
    )


def load_dataset(dataset_name: str):
    loaded_data = np.loadtxt(dataset_name, delimiter=",", dtype=object)
    cols_type = np.loadtxt(
        dataset_name[:-4] + "_cols_type.csv", delimiter=",", dtype=object
    )
    cols_type = np.array([Data.cols_type_maping[k] for k in cols_type])
    labels = loaded_data[1, :]
    loaded_data = loaded_data[1:, :]
    return loaded_data, cols_type, labels


def fill_na(data: np.array):
    data[data == ""] = -1
    return data


def fill_nan(data: np.array):
    data[np.isnan(data)] = -1
    return data


def load_sets():
    D_data = {}
    D_cols_type = {}
    D_labels = {}

    for file in listdir(os.path.abspath("datasets")):
        if (
            isfile(os.path.abspath("datasets") + "/" + file)
            and "_cols_type" not in file
        ):
            (
                D_data[file[:-4]],
                D_cols_type[file[:-4]],
                D_labels[file[:-4]],
            ) = load_dataset(os.path.abspath("datasets") + "/" + file)
    D = Data(D_data, D_cols_type, D_labels)
    return D


if __name__ == "__main__":

    D = load_sets()

    print(f"Loaded sets: {list(D.data.keys())}")

    ds1 = Dataset("adult", "cluster", "gower")
    ds2 = Dataset("adult", "cluster", "bin")
    ds3 = Dataset("adult", "cluster", "euclidean")
    ds4 = Dataset("adult", "cluster", "cosine")
    ds5 = Dataset("adult", "cluster", "minkowski")
    ds6 = Dataset("adult", "cluster", "dice")
    ds7 = Dataset("adult", "cluster", "jaccard")

    knn_test(ds1, D, 1000)
    knn_test(ds2, D, 1000)
    knn_test(ds3, D, 1000)
    knn_test(ds4, D, 1000)
    knn_test(ds5, D, 1000)
    knn_test(ds6, D, 1000)
    knn_test(ds7, D, 1000)

    # ================ For making comparison ================
    #
    # n_s = [50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250]
    #
    # results = np.zeros((len(n_s), 7, 2))
    #
    # for e, n in enumerate(n_s):
    #     print(f"{n}:")
    #     results[e][0][0], results[e][0][1] = knn_test(ds1, D, n)
    #     results[e][1][0], results[e][1][1] = knn_test(ds2, D, n)
    #     results[e][2][0], results[e][2][1] = knn_test(ds3, D, n)
    #     results[e][3][0], results[e][3][1] = knn_test(ds4, D, n)
    #     results[e][4][0], results[e][4][0] = knn_test(ds5, D, n)
    #     results[e][5][0], results[e][5][1] = knn_test(ds6, D, n)
    #     results[e][6][0], results[e][6][1] = knn_test(ds7, D, n)
    #
    # with open('hierarchical_clustering_res.txt', 'w') as f:
    #     for i in range(len(n_s)):
    #         f.write(str(i) + ':\n')
    #         for d in range(7):
    #             f.write(str(results[i][d][0]) + ' ' + str(results[i][d][1]) + '\n')
    #
